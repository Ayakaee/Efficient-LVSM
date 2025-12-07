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
import torch
from thop import profile
from utils.training_utils import format_number
import time

# Load config and read(override) arguments from CLI
config = init_config()
config.training.num_views = config.training.num_input_views + config.training.num_target_views
log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_eval.log')
logger = init_logging(log_file)

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

def print_gpu_memory():
    print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"显存峰值占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

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
if config.training.use_compile:
    model = torch.compile(model)
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

file = 'inc-none-ours-8.csv'
# with open(file, 'a', encoding='utf-8') as f:
    # f.write(','.join(['input', 'target', 'flops', 'memory', 'time']) + '\n')
    
# 如果启用特征提取，创建保存目录
if config.inference.extract_features and ddp_info.is_main_process:
    os.makedirs(config.inference.feature_save_dir, exist_ok=True)

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch_idx, batch in enumerate(dataloader):
        # print_gpu_memory()
        batch_o = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        for vi in range(config.vi, config.vi+1):
            for vt in range(config.vt, config.vt+1):
                torch.cuda.reset_peak_memory_stats()
                metrics = [str(vi), str(vt)]
                batch = batch_o.copy()
                model.module.config.training.num_input_views = vi
                model.module.config.training.num_target_views = vt
                model.module.config.training.num_views = vi + vt
                for k, v in batch.items():
                    if hasattr(v, 'shape'):
                        batch[k] = torch.repeat_interleave(v, vi+vt, dim=1)
                input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
                # flops, params = profile(model, inputs=(batch, input, target, False, False, False))
                # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs") # GFLOPs
                # print(f"参数量: {params / 1e6:.2f} M") # M (Million)
                for i in range(10):
                    result = model(batch, input, target, has_target_image=False, train=False)
                # metrics.append(f'{flops / 1e9:.2f}')
                # metrics.append(f'{torch.cuda.max_memory_allocated() / 1024**2:.2f}')
                # print_gpu_memory()
                
                tic = time.time()
                for i in range(50):
                    result = model(batch, input, target, has_target_image=False, train=False)
                print(f'推理用时：{(time.time() - tic) / 50}')
                metrics.append(f'{(time.time() - tic) / 50 * 1000:.2f}')
                with open(file, 'a', encoding='utf-8') as f:
                    f.write(','.join(metrics) + '\n')
        break


            

dist.barrier()
dist.destroy_process_group()
exit(0)