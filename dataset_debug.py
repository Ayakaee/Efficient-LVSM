# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging
from utils.metric_utils import export_results, summarize_evaluation, create_video_from_image_sequence
import argparse
import numpy as np
import shutil
from easydict import EasyDict as edict
from omegaconf import OmegaConf

# Load config and read(override) arguments from CLI
config = OmegaConf.load('configs/LVSM_ours.yaml')

# Convert to EasyDict if needed
config = OmegaConf.to_container(config, resolve=True)
config = edict(config)
config.inference.if_inference = True
config.training.dataset_path = 'data/test/partial_list.txt'
config.training.num_views = config.training.num_input_views + config.training.num_target_views
# log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_eval.log')
# logger = init_logging(log_file)

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
# ddp_info = init_distributed(seed=777)
# dist.barrier()

dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

# datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=12,
    shuffle=False,
    # num_workers=config.training.num_workers,
    # prefetch_factor=config.training.prefetch_factor,
    # persistent_workers=True,
    # pin_memory=False,
    # drop_last=True,
    # sampler=datasampler
)
dataloader_iter = iter(dataloader)

# dist.barrier()

# datasampler.set_epoch(0)
batch  = next(dataloader_iter)
print(len(dataset.all_scene_paths))

frames = []
from PIL import Image
path = '/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/junqiyou/processed_dataset/test/images/0196dedebec3dad2'
image_files = ['00028.png', '00082.png']
for image_file in image_files:
    img = Image.open(os.path.join(path, image_file))
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    # Convert to numpy array
    img.save(image_file)
create_video_from_image_sequence('/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/junqiyou/processed_dataset/test/images/03482c3bd66de195', 'data_video.mp4')
# dist.barrier()
# dist.destroy_process_group()
exit(0)