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
from thop import profile
from utils.training_utils import format_number
import time

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
file = 'inc-ours-8.csv'

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

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    model.module.clear_kv_cache()  # 清空缓存
    batch_o  = next(dataloader_iter)
    batch_o = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch_o.items()}
    # 使用增量推理
    vt = config.training.num_target_views
    for vi in range(1,config.training.num_input_views+1):
        print(vi)
        torch.cuda.reset_peak_memory_stats()
        metrics = [str(vi), str(vt)]
        batch = batch_o.copy()
        model.module.config.training.num_input_views = 1
        model.module.config.training.num_target_views = vt
        model.module.config.training.num_views = 1 + vt
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                batch[k] = torch.repeat_interleave(v, 1+vt, dim=1)
        input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
        # flops, params = profile(model, inputs=(batch, input, target, False, False, False, True))
        # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs") # GFLOPs
        # print(f"参数量: {params / 1e6:.2f} M") # M (Million)
        
        metrics.append('0')
        
        
        tic = time.time()
        for i in range(1):
            result = model(batch, input, target, has_target_image=False, train=False,use_incremental=True)
        metrics.append(f'{torch.cuda.max_memory_allocated() / 1024**2:.2f}')
        print(f'推理用时：{(time.time() - tic) / 1}')
        metrics.append(f'{(time.time() - tic) / 1 * 1000:.2f}')
        
        # print_gpu_memory()
        with open(file, 'a', encoding='utf-8') as f:
            f.write(','.join(metrics) + '\n')
   
dist.barrier()
dist.destroy_process_group()
exit(0)

