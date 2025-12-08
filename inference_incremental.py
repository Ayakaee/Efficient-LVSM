# Copyright (c) 2025 Yihang Sun. Efficient-LVSM project.
# 
# Incremental Inference Script: Process input views one by one, exporting results after each step
# 
# This implementation extends the LVSM framework by Haian Jin et al. (ICLR 2025)
# with incremental inference capabilities for improved efficiency.
# 
# Licensed under CC BY-NC-SA 4.0 - see LICENSE.md for details.

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging
from utils.metric_utils import export_results, summarize_evaluation
import argparse
import numpy as np
import shutil
from easydict import EasyDict as edict

# Load configuration and override parameters from command line
config = init_config()
config.training.num_views = config.training.num_input_views + config.training.num_target_views
config.uniform_views = False
log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_incremental_eval.log')
logger = init_logging(log_file)

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up distributed training/inference and fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up TF32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
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
    print(f"Running incremental inference; saving results to: {config.inference.checkpoint_dir}")
    # Avoid multiple processes downloading LPIPS at the same time
    import lpips
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()

datasampler.set_epoch(0)
model.eval()
print(f"Dataloader length: {len(dataloader)}")

# Create output directory
output_dir = os.path.join(config.inference.checkpoint_dir, 'incremental')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

if ddp_info.is_main_process:
    print(f"\n{'='*60}")
    print(f"Starting Incremental Inference Mode")
    print(f"Number of input views: {config.training.num_input_views}")
    print(f"Number of target views: {config.training.num_target_views}")
    print(f"Results will be exported after each incremental inference step")
    print(f"{'='*60}\n")

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    # Get a batch of data
    batch = next(dataloader_iter)
    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
    
    # Process data to get all input and target views
    input_all, target_all = model.module.process_data(
        batch, 
        has_target_image=True, 
        target_has_input=False, 
        compute_rays=True
    )
    
    # Clear KV cache
    model.module.clear_kv_cache()
    
    # Perform incremental inference for each view
    for view_idx in range(config.training.num_input_views):
        if ddp_info.is_main_process:
            print(f"\n{'='*50}")
            print(f"Processing input view {view_idx + 1}/{config.training.num_input_views}")
            print(f"{'='*50}")
        
        # Prepare input data for current view
        input_view = edict()
        input_view.image = input_all.image[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_o = input_all.ray_o[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_d = input_all.ray_d[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.c2w = input_all.c2w[:, view_idx:view_idx+1, :, :].clone()
        input_view.fxfycxcy = input_all.fxfycxcy[:, view_idx:view_idx+1, :].clone()
        input_view.index = input_all.index[:, view_idx:view_idx+1, :].clone()
        input_view.scene_name = input_all.scene_name
        
        # Prepare target view data (remains the same across iterations)
        target_view = edict()
        target_view.image = target_all.image.clone()
        target_view.ray_o = target_all.ray_o.clone()
        target_view.ray_d = target_all.ray_d.clone()
        target_view.image_h_w = target_all.image_h_w
        target_view.scene_name = target_all.scene_name
        target_view.index = target_all.index.clone()
        
        if ddp_info.is_main_process:
            print(f"Input view index: {input_view.index[0].cpu().numpy()}")
            print(f"Target view index: {target_view.index[0].cpu().numpy()}")
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Perform incremental inference
        result = model(
            batch, 
            input_view, 
            target_view, 
            train=False, 
            incremental_mode=True
        )
        
        # Print GPU memory peak
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            if ddp_info.is_main_process:
                print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # Render video if configured
        if config.inference.get("render_video", True):
            result = model.module.render_video(result, **config.inference.render_video_config)
        
        # Export results for current view
        view_output_dir = os.path.join(output_dir, f'view_{view_idx:03d}')
        os.makedirs(view_output_dir, exist_ok=True)
        
        if ddp_info.is_main_process:
            print(f"Exporting results to: {view_output_dir}")
        
        export_results(
            result, 
            view_output_dir, 
            compute_metrics=True, 
            resized=config.inference.get('resize', False),
            save_separate_images=True
        )
        
        if ddp_info.is_main_process:
            print(f"View {view_idx + 1} inference completed and exported")

dist.barrier()

if ddp_info.is_main_process:
    print(f"\n{'='*60}")
    print(f"All incremental inference completed!")
    print(f"Processed {config.training.num_input_views} input views in total")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}\n")
    
    if config.inference.get("compute_metrics", False):
        print("Computing evaluation metrics for each view...")
        for view_idx in range(config.training.num_input_views):
            view_output_dir = os.path.join(output_dir, f'view_{view_idx:03d}')
            print(f"\nEvaluation results for view {view_idx + 1}:")
            summarize_evaluation(view_output_dir)

dist.barrier()
dist.destroy_process_group()
exit(0)

