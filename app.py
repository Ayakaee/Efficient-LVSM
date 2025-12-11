# Copyright (c) 2025 Yihang Sun. Modifications for Efficient-LVSM.

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
from PIL import Image

# 引入项目模块
from setup import init_config

# 全局锁
inference_lock = threading.Lock()

# --- DDP 环境设置 ---
def setup_ddp_single_gpu():
    if dist.is_initialized():
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    backend = 'nccl' if torch.cuda.is_available() and sys.platform != 'win32' else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')

# --- 后端逻辑类 ---
class LVSMInteractiveBackend:
    def __init__(self):
        print("Initializing Efficient-LVSM Backend...")
        setup_ddp_single_gpu()
        
        self.config = init_config()
        self.config.training.num_views = self.config.training.num_input_views + self.config.training.num_target_views
             
        os.environ["OMP_NUM_THREADS"] = "1"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载数据集
        dataset_name = self.config.training.get("dataset_name", "data.dataset.Dataset")
        module_name, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module_name).__dict__[class_name]
        self.dataset = Dataset(self.config)
        
        # 2. 加载模型
        model_path = self.config.inference.checkpoint_dir.replace("evaluation", "checkpoints")
        module_name, class_name = self.config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module_name).__dict__[class_name]
        
        model = LVSM(self.config, logger=None).to(self.device)
        self.model = DDP(model, device_ids=[0], find_unused_parameters=False)
        
        if hasattr(self.model.module, "load_ckpt"):
            print(f"Loading checkpoint: {model_path}")
            self.model.module.load_ckpt(model_path)
        
        self.model.eval()
        self.scene_list = self._get_scene_names()
        
        self.current_scene_data = None
        self.current_scene_name = None

    def _get_scene_names(self):
        scenes = []
        max_scenes = min(len(self.dataset), 30)
        for i in range(max_scenes):
            try:
                path = self.dataset.all_scene_paths[i]
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                    scenes.append(data['scene_name'])
            except:
                scenes.append(f"Scene {i}")
        return scenes

    def load_scene_data(self, scene_name):
        if self.current_scene_name == scene_name and self.current_scene_data is not None:
            return self.current_scene_data

        try:
            scene_idx = self.scene_list.index(scene_name)
        except:
            print(f"Scene {scene_name} not found")
            return None

        batch = self.dataset[scene_idx]
        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                processed_batch[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, (str, list)):
                processed_batch[k] = [v]
            else:
                processed_batch[k] = v
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            input_all, target_all = self.model.module.process_data(
                processed_batch, 
                has_target_image=True, 
                target_has_input=False, 
                compute_rays=True
            )
        
        self.current_scene_data = (input_all, target_all)
        self.current_scene_name = scene_name
        return input_all, target_all

    def run_inference(self, input_slice, target_all, clear_cache=False):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            if clear_cache:
                self.model.module.clear_kv_cache()
            
            input_view = edict()
            input_view.image = input_slice.image.clone()
            input_view.ray_o = input_slice.ray_o.clone()
            input_view.ray_d = input_slice.ray_d.clone()
            input_view.c2w = input_slice.c2w.clone()
            input_view.fxfycxcy = input_slice.fxfycxcy.clone()
            input_view.index = input_slice.index.clone()
            input_view.scene_name = input_slice.scene_name
            
            target_view = edict()
            target_view.image = target_all.image.clone()
            target_view.ray_o = target_all.ray_o.clone()
            target_view.ray_d = target_all.ray_d.clone()
            target_view.image_h_w = target_all.image_h_w
            target_view.scene_name = target_all.scene_name
            target_view.index = target_all.index.clone()

            result = self.model(
                input_view,
                target_view,
                train=False,
                incremental_mode=True
            )
            
            render_img = result.render[0, 0].permute(1, 2, 0).cpu().float().numpy()
            render_img = (render_img * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(render_img)

# --- 初始化 Backend ---
backend = None
try:
    backend = LVSMInteractiveBackend()
except Exception as e:
    print(f"Error initializing backend: {e}")
    import traceback
    traceback.print_exc()

# --- 辅助函数：生成 Gallery 数据 ---
def get_input_gallery(input_all, count):
    """
    从 input_all 中提取前 count 帧，并转为 PIL List
    返回: [(Image, "View 1"), (Image, "View 2"), ...]
    """
    gallery_data = []
    for i in range(count):
        # [0, i] -> [C, H, W]
        img_tensor = input_all.image[0, i]
        img_np = img_tensor.permute(1, 2, 0).cpu().float().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        gallery_data.append((pil_img, f"Input View {i+1}"))
    return gallery_data

def tensor_to_pil(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).cpu().float().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

# --- Gradio 回调函数 ---

def on_scene_change(scene_name):
    if not backend: return None, [], None, 2, "Backend Error"
    
    with inference_lock:
        data = backend.load_scene_data(scene_name)
        if not data: return None, [], None, 2, "Data Error"
        input_all, target_all = data
        
        current_count = 2
        
        # 1. 准备推理数据
        input_slice = edict()
        for k, v in input_all.items():
            if isinstance(v, torch.Tensor) and k != 'scene_name':
                input_slice[k] = v[:, :current_count, ...]
            else:
                input_slice[k] = v
        
        # 2. 推理
        render_img = backend.run_inference(input_slice, target_all, clear_cache=True)
        
        # 3. 准备展示数据
        gt_img = tensor_to_pil(target_all.image[0, 0])
        input_gallery = get_input_gallery(input_all, current_count)
        
        log = f"Loaded {scene_name}. Reset to 2 views."
        return render_img, input_gallery, gt_img, current_count, log

def on_add_view(scene_name, current_count):
    if not backend: return None, [], None, current_count, "Backend Error"
    
    with inference_lock:
        if current_count >= 6:
            return gr.skip(), gr.skip(), gr.skip(), current_count, "Max views reached (6)."
        
        data = backend.load_scene_data(scene_name)
        input_all, target_all = data
        
        new_count = current_count + 1
        view_idx = current_count 
        
        # 1. 切出新增的那一帧
        input_slice = edict()
        for k, v in input_all.items():
            if isinstance(v, torch.Tensor) and k != 'scene_name':
                input_slice[k] = v[:, view_idx : view_idx+1, ...]
            else:
                input_slice[k] = v
        
        # 2. 增量推理
        render_img = backend.run_inference(input_slice, target_all, clear_cache=False)
        
        # 3. 准备展示数据
        gt_img = tensor_to_pil(target_all.image[0, 0])
        input_gallery = get_input_gallery(input_all, new_count)
        
        log = f"Added View {new_count}. Incremental inference."
        return render_img, input_gallery, gt_img, new_count, log

def on_remove_view(scene_name, current_count):
    if not backend: return None, [], None, current_count, "Backend Error"
    
    with inference_lock:
        if current_count <= 2:
            return gr.skip(), gr.skip(), gr.skip(), current_count, "Min views reached (2)."
        
        data = backend.load_scene_data(scene_name)
        input_all, target_all = data
        
        new_count = current_count - 1
        
        # 1. 切出前 N 帧
        input_slice = edict()
        for k, v in input_all.items():
            if isinstance(v, torch.Tensor) and k != 'scene_name':
                input_slice[k] = v[:, :new_count, ...]
            else:
                input_slice[k] = v
        
        # 2. 重新推理 (清空 Cache)
        render_img = backend.run_inference(input_slice, target_all, clear_cache=True)
        
        # 3. 准备展示数据
        gt_img = tensor_to_pil(target_all.image[0, 0])
        input_gallery = get_input_gallery(input_all, new_count)
        
        log = f"Removed View. Reset to {new_count} views."
        return render_img, input_gallery, gt_img, new_count, log

# --- UI 样式定义 ---
custom_css = """
#main-container { max-width: 1400px; margin: 0 auto; }
.view-btn { height: 50px; font-size: 16px; }
.stat-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; text-align: center; }
body, .gradio-container, .gradio-container * {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Arial, sans-serif !important;
}
"""

# --- 构建 Gradio UI ---
with gr.Blocks(title="Efficient-LVSM Interactive", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # 状态
    state_view_count = gr.State(value=2)

    # 标题栏
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>🚀 Efficient-LVSM: Incremental Novel View Synthesis</h1>
                <p>
                    This demo showcases the <b>incremental inference</b> capability. 
                    The model updates the target view prediction as you add more input views.
                </p>
            </div>
            """,
            elem_id="title_block",
        )

    with gr.Row(elem_id="main-container"):
        
        # --- 左侧控制栏 ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🎮 Controls")
            
            with gr.Group():
                scene_dropdown = gr.Dropdown(
                    label="1. Select Scene",
                    choices=backend.scene_list if backend else [],
                    value=backend.scene_list[0] if backend and backend.scene_list else None,
                    interactive=True
                )
                
                # 显眼的计数器
                with gr.Row(elem_classes="stat-box"):
                    view_indicator = gr.Number(
                        label="Active Input Views", 
                        value=2, 
                        interactive=False,
                        precision=0
                    )
                
                gr.Markdown("2. Adjust Views")
                with gr.Row():
                    btn_remove = gr.Button("➖ Remove View", variant="secondary", elem_classes="view-btn")
                    btn_add = gr.Button("➕ Add View", variant="primary", elem_classes="view-btn")
            
            # 日志区域
            gr.Markdown("### 📝 Logs")
            log_box = gr.Textbox(show_label=False, lines=12, max_lines=12, interactive=False, value="Ready.")

        # --- 右侧展示栏 ---
        with gr.Column(scale=3):
            
            # 1. 对比区域 (GT vs Prediction)
            gr.Markdown("### 👁️ Target View Comparison")
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        img_gt = gr.Image(label="Ground Truth (Target)", type="pil", interactive=False)
                    with gr.Column():
                        img_result = gr.Image(label="Model Prediction (Generated)", type="pil", interactive=False)

            # 2. 输入历史区域 (Gallery)
            gr.Markdown("### 🎞️ Accumulated Input Views")
            with gr.Group():
                gallery_inputs = gr.Gallery(
                    label="Input Sequence", 
                    show_label=False, 
                    columns=6, 
                    height="auto",
                    object_fit="contain",
                    preview=False
                )

    # --- 事件绑定 ---

    # 切换场景
    scene_dropdown.change(
        fn=on_scene_change,
        inputs=[scene_dropdown],
        outputs=[img_result, gallery_inputs, img_gt, state_view_count, log_box]
    ).then(lambda x: x, inputs=[state_view_count], outputs=[view_indicator])

    # 增加视角
    btn_add.click(
        fn=on_add_view,
        inputs=[scene_dropdown, state_view_count],
        outputs=[img_result, gallery_inputs, img_gt, state_view_count, log_box]
    ).then(lambda x: x, inputs=[state_view_count], outputs=[view_indicator])

    # 减少视角
    btn_remove.click(
        fn=on_remove_view,
        inputs=[scene_dropdown, state_view_count],
        outputs=[img_result, gallery_inputs, img_gt, state_view_count, log_box]
    ).then(lambda x: x, inputs=[state_view_count], outputs=[view_indicator])

    # 初始化加载
    demo.load(
        fn=on_scene_change,
        inputs=[scene_dropdown],
        outputs=[img_result, gallery_inputs, img_gt, state_view_count, log_box]
    )

if __name__ == "__main__":
    allowed_paths = [".", "/tmp", tempfile.gettempdir()]

    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        allowed_paths=allowed_paths,
        # root_path=""  # if running on servers, set to the specific root path
    )