# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

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

# Load config and read(override) arguments from CLI
config = init_config()
config.training.num_views = config.training.num_input_views + config.training.num_target_views
config.uniform_views = False
log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_eval.log')
logger = init_logging(log_file)

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
    # num_workers=config.training.num_workers,
    # prefetch_factor=config.training.prefetch_factor,
    # persistent_workers=True,
    # pin_memory=False,
    # drop_last=True,
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
    print(f"Running inference; save results to: {config.inference.checkpoint_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()
print(len(dataloader))

if os.path.exists(os.path.join(config.inference.checkpoint_dir, 'visualize')):
    shutil.rmtree(os.path.join(config.inference.checkpoint_dir, 'visualize'))
with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    use_incremental = config.inference.get("use_incremental_inference", False)
    
    if use_incremental:
        print(11111)
        model.module.clear_kv_cache()  # 清空缓存
        batch  = next(dataloader_iter)
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        input_all, target_all = model.module.process_data(batch, has_target_image=True, target_has_input = False, compute_rays=True)
        input = input_all
        target = target_all
        # 使用增量推理
        for vi in range(config.training.num_input_views):
            input_view = edict()
            print(vi)
            input_view.image = input_all.image[:, vi:vi+1, :, :, :].clone()
            input_view.ray_o = input_all.ray_o[:, vi:vi+1, :, :, :].clone()
            input_view.ray_d = input_all.ray_d[:, vi:vi+1, :, :, :].clone()
            input_view.c2w = input_all.c2w[:, vi:vi+1, :, :].clone()
            input_view.fxfycxcy = input_all.fxfycxcy[:, vi:vi+1, :].clone()
            input_view.index = input_all.index[:, vi:vi+1, :].clone()
            input_view.scene_name = input_all.scene_name
            # print('index: ', input_view.index)
            
            target_view = edict()
            target_view.image = target_all.image.clone()
            target_view.ray_o = target_all.ray_o.clone()
            target_view.ray_d = target_all.ray_d.clone()
            target_view.image_h_w = target_all.image_h_w
            target_view.scene_name = target_all.scene_name
            target_view.index = target_all.index.clone()
            print('index: ', target_view.index[1])
            print('ray0', input_view.ray_o[0][0][0][0][100:110])
            print('rayd', target_view.ray_d[0][0][0][0][100:110])
            result = model(batch, input_view, target_view, train=False, use_incremental=True)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Peak GPU Memory: {peak_memory:.2f} MB")
            
        if config.inference.get("render_video", True):
            result= model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference.checkpoint_dir, compute_metrics=True, incremental=vi)
    else:
        # if True:
        for batch_idx, batch in enumerate(dataloader):
            print(23456)
            # if batch_idx < 16:
                # continue
            # if batch_idx > 32:
                # break
            # batch = dataset.__getitem__(88)
            # batch = {k: torch.unsqueeze(v.to(ddp_info.device), 0) if type(v) == torch.Tensor else v for k, v in batch.items()}
            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
            input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
            print('index: ', input.index[0])
            result = model(batch, input, target, train=False)
                
            if config.inference.get("render_video", True):
                result= model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference.checkpoint_dir, compute_metrics=True, resized=config.inference.get('resize', False), save_separate_images=True)
            
dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference.checkpoint_dir)
    # if config.inference.get("generate_website", True):
    #     os.system(f"python generate_html.py {config.inference.checkpoint_dir}")

dist.barrier()
dist.destroy_process_group()
exit(0)

