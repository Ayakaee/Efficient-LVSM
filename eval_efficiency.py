# Copyright (c) 2025 Haian Jin. Original LVSM implementation (ICLR 2025).
# Copyright (c) 2025 Yihang Sun. Modifications for Efficient-LVSM.
#
# This code is based on the LVSM project by Haian Jin et al.
# Original repository: https://github.com/Haian-Jin/LVSM
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

# Set up DDP training/inference and fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    "tf32": torch.float32,
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
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

dist.barrier()

# Import model and load checkpoint
model_path = config.inference.checkpoint_dir
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
# model.module.load_ckpt(model_path)

file = "inc-ours-8.csv"
with open(file, "w", encoding="utf-8") as f:
    f.write(",".join(["v_input", "v_target", "memory", "time"]) + "\n")

if ddp_info.is_main_process:
    print(f"[INFO] Running inference; save results to: {config.inference.checkpoint_dir}")
    print(f"[INFO] num_input_views={config.training.num_input_views}, "
          f"num_target_views={config.training.num_target_views}")
    print(f"[INFO] Using AMP={config.training.use_amp}, dtype={config.training.amp_dtype}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

dist.barrier()

datasampler.set_epoch(0)
model.eval()

# Number of full incremental runs to average over
n_repeat = 20
vt = config.training.num_target_views
max_vi = config.training.num_input_views

# Record multiple measurements of memory / time for each vi
mem_stats = {vi: [] for vi in range(1, max_vi + 1)}
time_stats = {vi: [] for vi in range(1, max_vi + 1)}

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    # Take a single batch and reuse it across all experiments for fair comparison
    batch_o = next(dataloader_iter)
    batch_o = {
        k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_o.items()
    }

    if ddp_info.is_main_process:
        print(f"[INFO] Starting measurement with {n_repeat} full incremental runs...")

    for rep in range(n_repeat):
        # Clear incremental cache at the beginning of each full run
        model.module.clear_kv_cache()

        if ddp_info.is_main_process:
            print(f"[REP {rep + 1}/{n_repeat}] Clearing KV cache and running incremental inference...")

        for vi in range(1, max_vi + 1):
            torch.cuda.reset_peak_memory_stats()

            batch = batch_o.copy()

            # Keep your original training config logic
            model.module.config.training.num_input_views = 1
            model.module.config.training.num_target_views = vt
            model.module.config.training.num_views = 1 + vt

            for k, v in batch.items():
                if hasattr(v, "shape"):
                    batch[k] = torch.repeat_interleave(v, 1 + vt, dim=1)

            input, target = model.module.process_data(
                batch,
                has_target_image=True,
                target_has_input=config.training.target_has_input,
                compute_rays=True,
            )

            # Synchronize to make sure kernels are included in timing
            torch.cuda.synchronize(ddp_info.device)
            tic = time.time()

            result = model(
                input,
                target,
                has_target_image=False,
                train=False,
                incremental_mode=True,
            )

            torch.cuda.synchronize(ddp_info.device)

            max_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            elapsed_ms = (time.time() - tic) * 1000.0

            mem_stats[vi].append(max_mem_mb)
            time_stats[vi].append(elapsed_ms)

            if ddp_info.is_main_process:
                print(f"[REP {rep + 1}/{n_repeat}] vi={vi}/{max_vi}: "
                      f"peak_mem={max_mem_mb:.2f} MB, time={elapsed_ms:.2f} ms")


if ddp_info.is_main_process:
    file = "incremental_result.csv"
    with open(file, "w", encoding="utf-8") as f:
        f.write(",".join(["v_input", "v_target", "memory", "time"]) + "\n")
        print("[INFO] Writing averaged results to CSV:")
        for vi in range(1, max_vi + 1):
            mem_mean = np.mean(mem_stats[vi])
            time_mean = np.mean(time_stats[vi])
            f.write(f"{vi},{vt},{mem_mean:.2f},{time_mean:.2f}\n")
            print(f"  vi={vi}: avg_peak_mem={mem_mean:.2f} MB, avg_time={time_mean:.2f} ms")

dist.barrier()
dist.destroy_process_group()
exit(0)
