import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gradio as gr
import numpy as np
from easydict import EasyDict as edict
import threading
import importlib

# 引入项目模块
from setup import init_config
# 注意：我们这里手动初始化 distributed，不使用 setup.py 里的 init_distributed 以避免参数解析冲突

# 锁，防止多用户同时推理导致KV Cache冲突
inference_lock = threading.Lock()

def setup_ddp_single_gpu():
    """配置单卡DDP环境"""
    if not dist.is_initialized():
        # 设置伪分布式环境，World Size = 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # 随便选一个空闲端口
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # 初始化进程组
        # nccl backend 是 GPU 训练的标准，但在某些非Linux环境可能需要 'gloo'
        backend = 'nccl' if torch.cuda.is_available() and sys.platform != 'win32' else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        print(f"DDP Initialized: Backend={backend}, World Size=1, Rank=0")

class LVSMIncrementalDemo:
    def __init__(self):
        print("Initializing Efficient-LVSM Demo with DDP...")
        
        # 1. 设置 DDP 环境
        setup_ddp_single_gpu()
        
        # 2. 初始化配置
        self.config = init_config()
        self.config.training.num_views = self.config.training.num_input_views + self.config.training.num_target_views
        
        # 强制设置线程数和显卡
        os.environ["OMP_NUM_THREADS"] = "1"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 3. 加载数据集
        dataset_name = self.config.training.get("dataset_name", "data.dataset.Dataset")
        module_name, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module_name).__dict__[class_name]
        
        self.dataset = Dataset(self.config)
        print(f"Dataset loaded. Total scenes: {len(self.dataset)}")
        
        # 4. 加载模型
        model_path = self.config.inference.checkpoint_dir.replace("evaluation", "checkpoints")
        module_name, class_name = self.config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module_name).__dict__[class_name]
        
        # 实例化模型
        model = LVSM(self.config, logger=None).to(self.device)
        
        # --- 关键修改：DDP 包装 ---
        # device_ids=[0] 对应当前进程使用的显卡
        self.model = DDP(model, device_ids=[0], find_unused_parameters=False)
        
        # 加载权重 (注意：现在通过 self.model.module 调用 load_ckpt)
        if hasattr(self.model.module, "load_ckpt"):
            print(f"Loading checkpoint from: {model_path}")
            self.model.module.load_ckpt(model_path)
        else:
            print("Warning: load_ckpt method not found on model.module")
            
        self.model.eval()
        
        # 缓存场景列表
        self.scene_list = self._get_scene_names()

    def _get_scene_names(self):
        scenes = []
        # 为了演示快速加载，只取前50个，你可以去掉切片
        max_scenes = min(len(self.dataset), 20)
        for i in range(max_scenes):
            try:
                # 简单读取 scene_name
                path = self.dataset.all_scene_paths[i]
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                    scenes.append(data['scene_name'])
            except:
                scenes.append(f"Scene {i}")
        return scenes

    def run_incremental_inference(self, scene_name):
        with inference_lock:
            # 1. 找到对应的 scene_idx
            try:
                scene_idx = self.scene_list.index(scene_name)
            except ValueError:
                return [], "Scene not found error.", None

            # 2. 获取数据
            batch = self.dataset[scene_idx]
            
            # 增加 Batch 维度并移动到 GPU
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.unsqueeze(0).to(self.device)
                elif isinstance(v, (str, list)):
                    processed_batch[k] = [v]
                else:
                    processed_batch[k] = v
            
            # 3. 数据预处理
            # 注意：调用 model.module.process_data
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                input_all, target_all = self.model.module.process_data(
                    processed_batch, 
                    has_target_image=True, 
                    target_has_input=False, 
                    compute_rays=True
                )

                # 4. 清空 KV Cache
                # 注意：调用 model.module.clear_kv_cache
                self.model.module.clear_kv_cache()
                
                v_input = input_all.image.shape[1]
                results_gallery = []
                
                # 获取 Target GT
                target_gt = target_all.image[0, 0].permute(1, 2, 0).cpu().numpy()
                target_gt = (target_gt * 255).clip(0, 255).astype(np.uint8)
                
                log_text = f"Processing Scene: {scene_name}\nTarget GT Loaded.\n"
                # 初始化 yielded 变量
                step_visual = None 
                yield results_gallery, log_text, step_visual

                # 5. 增量循环
                for view_idx in range(v_input):
                    log_text += f"Step {view_idx + 1}/{v_input}: Processing input view...\n"
                    
                    # 准备当前视角的输入数据
                    input_view = edict()
                    input_view.image = input_all.image[:, view_idx:view_idx+1, ...].clone()
                    input_view.ray_o = input_all.ray_o[:, view_idx:view_idx+1, ...].clone()
                    input_view.ray_d = input_all.ray_d[:, view_idx:view_idx+1, ...].clone()
                    input_view.c2w = input_all.c2w[:, view_idx:view_idx+1, ...].clone()
                    input_view.fxfycxcy = input_all.fxfycxcy[:, view_idx:view_idx+1, ...].clone()
                    input_view.index = input_all.index[:, view_idx:view_idx+1, ...].clone()
                    input_view.scene_name = input_all.scene_name
                    
                    target_view = edict()
                    target_view.image = target_all.image.clone()
                    target_view.ray_o = target_all.ray_o.clone()
                    target_view.ray_d = target_all.ray_d.clone()
                    target_view.image_h_w = target_all.image_h_w
                    target_view.scene_name = target_all.scene_name
                    target_view.index = target_all.index.clone()

                    # 执行推理
                    # 注意：直接调用 self.model()，DDP 会处理 forward
                    result = self.model(
                        input_view,
                        target_view,
                        train=False,
                        incremental_mode=True
                    )
                    
                    # 后处理
                    render_img = result.render[0, 0].permute(1, 2, 0).cpu().float().numpy()
                    render_img = (render_img * 255).clip(0, 255).astype(np.uint8)
                    
                    curr_input_img = input_view.image[0, 0].permute(1, 2, 0).cpu().numpy()
                    curr_input_img = (curr_input_img * 255).clip(0, 255).astype(np.uint8)

                    caption = f"Step {view_idx+1}"
                    results_gallery.append((render_img, caption))
                    
                    step_visual = [
                        (curr_input_img, f"Input View {view_idx+1}"),
                        (render_img, f"Prediction (Inputs: {view_idx+1})")
                    ]

                    yield results_gallery, log_text, step_visual
                
                log_text += "Inference Completed!"
                
                step_visual = [
                    (render_img, "Final Prediction"),
                    (target_gt, "Ground Truth")
                ]
                yield results_gallery, log_text, step_visual

# --- 初始化与启动 ---
try:
    demo_backend = LVSMIncrementalDemo()
    is_loaded = True
except Exception as e:
    print(f"Failed to initialize demo backend: {e}")
    import traceback
    traceback.print_exc()
    is_loaded = False
    demo_backend = None

with gr.Blocks(title="Efficient-LVSM Incremental (DDP Mode)") as demo:
    gr.Markdown("# Efficient-LVSM: Incremental Inference (DDP Wrapper)")
    gr.Markdown("此演示在单卡上模拟 DDP 环境 (World Size=1)，保持了原始代码结构。")
    
    with gr.Row():
        with gr.Column(scale=1):
            if is_loaded:
                scene_dropdown = gr.Dropdown(
                    choices=demo_backend.scene_list, 
                    label="Select Scene", 
                    value=demo_backend.scene_list[0] if demo_backend.scene_list else None
                )
            else:
                scene_dropdown = gr.Dropdown(label="Select Scene (Model Failed to Load)", choices=[])
                
            run_btn = gr.Button("Start Inference", variant="primary")
            logs = gr.Textbox(label="Logs", interactive=False, lines=10)
        
        with gr.Column(scale=2):
            gr.Markdown("### Current Step")
            current_step_gallery = gr.Gallery(label="Input vs Prediction", columns=2, height=400, object_fit="contain")
            
            gr.Markdown("### Progression")
            progression_gallery = gr.Gallery(label="History", columns=4, object_fit="contain")

    run_btn.click(
        fn=demo_backend.run_incremental_inference,
        inputs=[scene_dropdown],
        outputs=[progression_gallery, logs, current_step_gallery]
    )

if __name__ == "__main__":
    # cleanup 在脚本退出时可能需要，但在 gradio serve 中通常由手动关闭触发
    demo.queue().launch(share=False, server_name="0.0.0.0")