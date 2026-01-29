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
import csv
import shutil

# Load config and read(override) arguments from CLI
config = init_config()
config.inference.extract_features = False
config.training.num_views = config.training.num_input_views + config.training.num_target_views
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

if config.inference.extract_features and ddp_info.is_main_process:
    os.makedirs(config.inference.feature_save_dir, exist_ok=True)

st, end = (0,20000)
step = 1
print(model_path)
print(os.listdir(model_path))
# models = [name for name in os.listdir(model_path) if 'ckpt_t' not in name and 'ckpt' in name and st <= int(name.split('_')[1].split('.')[0]) // 100 <= end]
models = [name for name in os.listdir(model_path) if 'ckpt_t' in name and st <= int(name.split('t')[2].split('.')[0]) <= end]
models.sort(key=lambda x: float(x.split('t')[2].split('h')[0]))
models = models[::step]
print(models)
print(len(dataloader))
metric_file = os.path.join(model_path, 'ametrics-eval.csv')
print(metric_file)
with open(metric_file, 'a', encoding='utf-8') as f:
    f.write(','.join(['model', 'psnr', 'lpips', 'ssim']) + '\n')

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for i, name in enumerate(models):
        # if i == 0:
        del model
        torch.cuda.empty_cache()
        
        LVSM = importlib.import_module(module).__dict__[class_name]
        model = LVSM(config, logger).to(ddp_info.device)
        model = DDP(model, device_ids=[ddp_info.local_rank])
        model.module.load_ckpt(os.path.join(model_path, name))
        model.eval()
        if os.path.exists(os.path.join(config.inference.checkpoint_dir, 'visualize')):
            shutil.rmtree(os.path.join(config.inference.checkpoint_dir, 'visualize'))
        datasampler.set_epoch(0)
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
            input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
            
            result = model(batch, input, target, train=False)
                
            if config.inference.get("render_video", False):
                result= model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference.checkpoint_dir, compute_metrics=config.inference.get("compute_metrics"), resized=config.inference.get('resize', False))

        metrics = summarize_evaluation(config.inference.checkpoint_dir)
        metrics.insert(0, str(float(name.split('t')[2].split('h')[0])))
        with open(metric_file, 'a', encoding='utf-8') as f:
            f.write(','.join(metrics) + '\n')
        
        # if config.inference.get("generate_website", True):
        #     os.system(f"python generate_html.py {config.inference.checkpoint_dir}")

dist.barrier()
dist.destroy_process_group()
exit(0)