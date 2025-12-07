# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Demo inference script with camera pose manipulation

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging
from utils.metric_utils import export_results
import argparse
import numpy as np
import shutil
from easydict import EasyDict as edict
import cv2
from PIL import Image
from utils.data_utils import ProcessData

def rotate_camera_batch(c2w_batch, axis='y', angle_degrees=0):
    """
    批量旋转相机姿态
    Args:
        c2w_batch: 相机到世界的变换矩阵 [B, V, 4, 4]
        axis: 旋转轴 ('x', 'y', 'z')
        angle_degrees: 旋转角度（度）
    Returns:
        新的c2w矩阵 [B, V, 4, 4]
    """
    angle_rad = np.deg2rad(angle_degrees)
    
    if axis == 'x':
        rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif axis == 'y':
        rot = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif axis == 'z':
        rot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    # 应用旋转 [B, V, 4, 4] @ [4, 4] = [B, V, 4, 4]
    rot_tensor = torch.from_numpy(rot).to(c2w_batch.device)
    c2w_new = torch.matmul(c2w_batch, rot_tensor)
    return c2w_new


def translate_camera_batch(c2w_batch, translation_vector):
    """
    批量平移相机位置
    Args:
        c2w_batch: 相机到世界的变换矩阵 [B, V, 4, 4]
        translation_vector: 平移向量 [x, y, z]
    Returns:
        新的c2w矩阵 [B, V, 4, 4]
    """
    c2w_new = c2w_batch.clone()
    translation = torch.tensor(translation_vector, dtype=c2w_batch.dtype, device=c2w_batch.device)
    c2w_new[:, :, :3, 3] += translation
    return c2w_new


def apply_camera_transform(target_view, transform_config, device):
    """
    对target view应用相机变换
    Args:
        target_view: 目标view的数据
        transform_config: 变换配置，包含旋转和平移参数
        device: torch device
    Returns:
        新的target_view
    """
    # 复制target_view
    new_target = edict()
    new_target['name'] = []
    print(target_view.keys())
    for key in target_view.keys():
        if isinstance(target_view[key], torch.Tensor):
            new_target[key] = target_view[key].clone()
        else:
            new_target[key] = target_view[key]
    
    batch_size = target_view.c2w.shape[0]
    num_views = target_view.c2w.shape[1]
    
    c2w = target_view.c2w.clone()  # [B, V, 4, 4]
    
    # 构建文件名
    name_suffix = ""
    
    # 应用旋转
    if 'rotate_x' in transform_config:
        c2w = rotate_camera_batch(c2w, axis='x', angle_degrees=transform_config.rotate_x)
        name_suffix += f"_rx{transform_config.rotate_x}"
    if 'rotate_y' in transform_config:
        c2w = rotate_camera_batch(c2w, axis='y', angle_degrees=transform_config.rotate_y)
        name_suffix += f"_ry{transform_config.rotate_y}"
    if 'rotate_z' in transform_config:
        c2w = rotate_camera_batch(c2w, axis='z', angle_degrees=transform_config.rotate_z)
        name_suffix += f"_rz{transform_config.rotate_z}"
    
    # 应用平移
    if 'translate' in transform_config:
        c2w = translate_camera_batch(c2w, transform_config.translate)
        name_suffix += f"_tx{transform_config.translate[0]}_ty{transform_config.translate[1]}_tz{transform_config.translate[2]}"
    
    # 更新c2w
    new_target.c2w = c2w
    
    image_height, image_width = new_target["image_h_w"]
    ray_o, ray_d = model.module.process_data.compute_rays(new_target['c2w'], new_target['fxfycxcy'], 
                                        image_height, image_width, device=target_view["ray_o"].device)
    new_target["ray_o"], new_target["ray_d"] = ray_o, ray_d
    
    # 更新名称（如果存在）
    if hasattr(new_target, 'name') and isinstance(new_target.name, list):
        for v in range(num_views):
            idx = new_target['index'][0][v][1]
            new_target.name.append(f'{idx.item():03d}{name_suffix}')
    
    print('ok')
    return new_target


config = init_config()
config.training.num_views = config.training.num_input_views + config.training.num_target_views
log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_demo.log')
logger = init_logging(log_file)
args = config.demo
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()

# Import model and load checkpoint
model_path = config.inference.checkpoint_dir.replace("evaluation", "checkpoints")
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(model_path)

if ddp_info.is_main_process:  
    print(f"Running demo inference; save results to: {args.output_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

datasampler.set_epoch(0)
model.eval()

# 准备相机变换配置

import json
import numpy as np

def export_camera_json(out_list, target_view, transform_name="center", delta_pitch=0.0, delta_yaw=0.0):
    """
    将 target_view 的所有 view 导出并追加到 out_list 中。
    Args:
        out_list: list, 已有的输出项列表，会在函数内被追加并返回
        target_view: 包含 .c2w, .fxfycxcy, .index, 可选 .name 的结构（batch 第一维为0）
        transform_name: str, 比如 "center", "up", "down", "left", "right"
        delta_pitch: float, 相对中心的俯仰（度）
        delta_yaw: float, 相对中心的偏航（度）
    Returns:
        out_list: 被追加后的列表
    """
    # 支持 target_view 中的 tensor / numpy 混合情况
    # 取 batch=0
    c2w = target_view.c2w[0].cpu().numpy()           # [V,4,4]
    fxfycxcy = target_view.fxfycxcy[0].cpu().numpy() # [V,4]
    index = target_view.index[0].cpu().numpy()       # [V,2] (scene_idx, view_idx)

    # 如果 target_view 有 name（list），读取它；否则用 index 拼 name
    names = None
    if hasattr(target_view, "name") and isinstance(target_view.name, (list, tuple)):
        # name 在 apply_camera_transform 中被填为 list of strings
        names = target_view.name

    num_views = c2w.shape[0]
    print(num_views)
    for v in range(num_views):
        M = c2w[v]    # 4x4
        # R = M[:3, :3]
        # t = M[:3, 3]
        R_c2w = M[:3,:3]
        t_c2w = M[:3, 3]

        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        R = R_w2c
        t = t_w2c
        fx, fy, cx, cy = fxfycxcy[v].tolist()

        view_id = v
        # view_id = int(index[v][1])

        view_name = None
        if names and v < len(names):
            # 如果 names 中包含像 "012_rx10" 的字符串，更好被保留
            try:
                view_name = str(names[v])
            except:
                view_name = f"view_{view_id:03d}"
        else:
            view_name = f"view_{view_id:03d}"

        view_info = {
            "transform_name": transform_name,     # which transform produced this pose
            "delta": {"pitch": float(delta_pitch), "yaw": float(delta_yaw)},
            # "scene_id": scene_id,
            "view_id": view_id,
            "name": view_name,
            "R": R.tolist(),
            "t": t.tolist(),
            "c2w": M.tolist(),
            "position": t.tolist(),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy)
        }

        out_list.append(view_info)

    return out_list

transform_config = edict()
if args.rotate_x != 0:
    transform_config.rotate_x = args.rotate_x
if args.rotate_y != 0:
    transform_config.rotate_y = args.rotate_y
if args.rotate_z != 0:
    transform_config.rotate_z = args.rotate_z
if args.translate_x != 0 or args.translate_y != 0 or args.translate_z != 0:
    transform_config.translate = [args.translate_x, args.translate_y, args.translate_z]

if ddp_info.is_main_process:
    print("="*50)
    print("Demo 配置:")
    # print(f"  基准view索引: {args.base_view_idx}")
    print(f"  旋转 (X, Y, Z): ({args.rotate_x}°, {args.rotate_y}°, {args.rotate_z}°)")
    print(f"  平移 (X, Y, Z): ({args.translate_x}, {args.translate_y}, {args.translate_z})")
    print("="*50)

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping["fp32"],
):
    # 只处理一个batch作为demo
    batch = next(dataloader_iter)
    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
    
    # 处理数据
    input, target = model.module.process_data(
        batch, 
        has_target_image=False,  # Demo模式不需要ground truth
        target_has_input=config.training.target_has_input, 
        compute_rays=True
    )
    
    if ddp_info.is_main_process:
        print(f"场景名称: {input.scene_name}")
        print(f"Input views 索引: {input.index[0]}")
        # print(f"Target views 索引: {target.index[0]}")
        print(f"原始相机位置: {target.c2w[0, 0, :3, 3].cpu().numpy()}")
    
    out = []

    theta = 10.0

    # 1) center (原始 target)
    out = export_camera_json(out, target, transform_name="center", delta_pitch=0.0, delta_yaw=0.0)

    # 2) up
    transform_config.rotate_x = theta
    transform_config.rotate_y = 0
    transform_config.rotate_z = 0
    transform_config.translate = [0,0,0]
    target_transformed = apply_camera_transform(target, transform_config, ddp_info.device)
    out = export_camera_json(out, target_transformed, transform_name="up", delta_pitch=theta, delta_yaw=0.0)

    # 3) down
    transform_config.rotate_x = -theta
    transform_config.rotate_y = 0
    transform_config.rotate_z = 0
    transform_config.translate = [0,0,0]
    target_transformed = apply_camera_transform(target, transform_config, ddp_info.device)
    out = export_camera_json(out, target_transformed, transform_name="down", delta_pitch=-theta, delta_yaw=0.0)

    # 4) left
    transform_config.rotate_x = 0
    transform_config.rotate_y = theta
    transform_config.rotate_z = 0
    transform_config.translate = [0,0,0]
    target_transformed = apply_camera_transform(target, transform_config, ddp_info.device)
    out = export_camera_json(out, target_transformed, transform_name="left", delta_pitch=0.0, delta_yaw=theta)

    # 5) right
    transform_config.rotate_x = 0
    transform_config.rotate_y = -theta
    transform_config.rotate_z = 0
    transform_config.translate = [0,0,0]
    target_transformed = apply_camera_transform(target, transform_config, ddp_info.device)
    out = export_camera_json(out, target_transformed, transform_name="right", delta_pitch=0.0, delta_yaw=-theta)

    # 写入一个单一 JSON 文件（此文件包含所有 view 的所有变换）
    output_json = {
        "scene_name": input.scene_name if hasattr(input, "scene_name") else None,
        "base_views": [int(i[1]) for i in target.index[0].cpu().numpy()],  # 所有原始 view id 列表
        "cameras": out
    }
    filename = 'camera-w2c.json'
    with open(filename, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"Saved merged camera JSON: {filename}")

dist.barrier()
dist.destroy_process_group()
print("Demo完成！")

