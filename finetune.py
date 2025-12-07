# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup, init_logging, init_file_logging
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from model.encoder import load_encoders, preprocess_raw_image
from einops import rearrange, repeat
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from datetime import datetime

def load_pretrained_checkpoint(model, checkpoint_path, config):
    """
    Load pretrained checkpoint from 256 resolution training and adapt to 512 resolution.
    """
    print_rank0(f"Loading pretrained checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu",  weights_only=True)
    state_dict = checkpoint["model"]
    if not config.training.use_compile:
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    else:
        state_dict = {k if k.startswith('_orig_mod.') else '_orig_mod.' + k: v for k, v in state_dict.items()}
    print(state_dict.keys())
    # Load model state dict with strict=False to handle resolution changes
    if isinstance(model, DDP):
        missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print_rank0(f"Loaded checkpoint successfully")
    print_rank0(f"Missing keys: {len(missing_keys)}")
    print_rank0(f"Missing keys: {missing_keys}")
    print_rank0(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Print some missing/unexpected keys for debugging
    if missing_keys:
        print_rank0(f"First few missing keys: {missing_keys[:5]}")
    if unexpected_keys:
        print_rank0(f"First few unexpected keys: {unexpected_keys[:5]}")
    
    return model

def create_finetune_config(base_config, pretrained_checkpoint_path):
    """
    Create finetune configuration based on the paper's requirements:
    - Resolution: 512
    - Learning rate: 1e-4
    - Total batch size: 128
    - Training steps: 20k iterations
    """
    # Create a copy of the base config while maintaining EasyDict structure
    from easydict import EasyDict
    finetune_config = EasyDict(base_config.copy())
    
    # Update resolution-related settings
    if base_config.training.get('high_resolution', True):
        finetune_config.model.image_tokenizer.image_size = 512
        finetune_config.model.target_pose_tokenizer.image_size = 512
    
    # Update training parameters for finetuning
    finetune_config.training.lr = 1e-4  # Smaller learning rate for finetuning
    # finetune_config.training.train_steps = 20000  # 20k iterations
    # finetune_config.training.batch_size_per_gpu = 16  # Adjust based on your GPU memory
    finetune_config.training.grad_accum_steps = 1  # Adjust to get total batch size of 128
    
    # Update checkpoint and logging settings
    # finetune_config.training.checkpoint_dir = f"./experiments/checkpoints/finetune_512"
    # finetune_config.training.wandb_exp_name = f"{base_config.training.wandb_exp_name}_finetune_512"
    
    # Set pretrained checkpoint path
    finetune_config.training.pretrained_checkpoint = pretrained_checkpoint_path
    
    # Update dataset path for 512 resolution (if different)
    # finetune_config.training.dataset_path = "re10k/train/full_list_512.txt"  # Uncomment if you have 512 data
    
    return finetune_config

def main():
    # Load base configuration
    config = init_config()
    
    # Check if pretrained checkpoint path is provided
    if not hasattr(config.training, 'pretrained_checkpoint') or not config.training.pretrained_checkpoint:
        raise ValueError("Please provide pretrained_checkpoint path in config or command line")
    
    pretrained_checkpoint_path = config.training.pretrained_checkpoint
    
    # Create finetune configuration
    config = create_finetune_config(config, pretrained_checkpoint_path)
    
    # Update number of views
    config.training.num_views = config.training.num_input_views + config.training.num_target_views
    
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    
    # Set up logging
    log_file = config.training.get("log_file", f'logs/{config.training.wandb_exp_name}')
    logger = init_logging(log_file)
    file_only_logger = init_file_logging(log_file)
    
    # Set up DDP for training/inference and Fix random seed
    ddp_info = init_distributed(seed=777)
    print(f"Rank Available memory before barrier: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"Rank  Allocated memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    print(f"Rank  Cached memory: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    dist.barrier()
    
    # Set up wandb and backup source code
    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()
    
    

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
    config.training.dataset_len = len(dataset)
    
    # Calculate training parameters
    total_num_epochs = config.training.train_epochs
    grad_accum_steps = config.training.grad_accum_steps
    batch_size_per_gpu = config.training.batch_size_per_gpu
    total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
    total_train_steps = int(total_num_epochs * config.training.dataset_len // (total_batch_size))
    total_param_update_steps = total_train_steps
    total_train_steps = total_train_steps * grad_accum_steps
    save_every_steps = int(config.training.dataset_len // (total_batch_size) * config.training.checkpoint_every_epoch) + 1
    
    print_rank0(f"Finetuning configuration:")
    print_rank0(f"  Resolution: 512x512")
    print_rank0(f"  Learning rate: {config.training.lr}")
    print_rank0(f"  Total batch size: {total_batch_size}")
    print_rank0(f"  Training steps: {total_train_steps}")
    print_rank0(f"  Pretrained checkpoint: {pretrained_checkpoint_path}")
    
    # Record training start time
    training_start_time = datetime.now()
    total_train_hours = config.training.get("train_time", 12)
    save_every_hours = config.training.get("checkpoint_every_time", 1.0)
    save_every_seconds = int(save_every_hours * 3600)
    total_train_seconds = int(total_train_hours * 3600)
    last_time_save = training_start_time
    
    datasampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=config.training.prefetch_factor,
        sampler=datasampler,
    )
    dataloader_iter = iter(dataloader)
    
    # Load model
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config, logger).to(ddp_info.device)
    
    if config.training.use_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=False)
    
    # Load pretrained checkpoint
    model = load_pretrained_checkpoint(model, pretrained_checkpoint_path, config)
    
    # Create optimizer and scheduler for finetuning
    optimizer, optimized_param_dict, all_param_dict = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    optim_param_list = list(optimized_param_dict.values())
    
    scheduler_type = config.training.get("scheduler_type", "cosine")
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_param_update_steps,  # Use finetune steps instead of total steps
        config.training.warmup,
        scheduler_type=scheduler_type,
    )
    
    # Set up gradient scaler
    enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
    print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
    dist.barrier()
    
    # Initialize training state
    start_train_step = 0
    cur_param_update_step = 0
    model.train()
    
    print_rank0(f"Starting finetuning from step {start_train_step}")
    print_rank0(f"Dataset length: {len(dataset)}")
    
    # Training loop
    cur_train_step = start_train_step
    cur_param_update_step = 0
    
    while cur_train_step <= total_train_steps and (datetime.now() - training_start_time).total_seconds() < total_train_seconds:
        tic = time.time()
        cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
        try:
            data = next(dataloader_iter)
        except StopIteration:
            print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
            datasampler.set_epoch(cur_epoch)
            dataloader_iter = iter(dataloader)
            try:
                data = next(dataloader_iter)
            except Exception as e:
                logger.warning(f"DataLoader error after reset: {e}, skipping batch")
                continue
        except Exception as e:
            logger.warning(f"DataLoader error: {e}, skipping batch")
            continue
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}
        data_time = time.time() - tic
        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
            ret_dict = model(batch, input, target)

        current_epoch = int(cur_train_step * total_batch_size // len(dataset))
        if current_epoch > config.training.repa_stop_epoch and config.training.repa_stop_epoch > -1:
            model.module.loss_computer.config.training.proj_loss_weight = 0.0
        update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
        if update_grads:
            with model.no_sync(): # no sync grads for efficiency
                scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        else:
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        cur_train_step += 1

        export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if update_grads and (not skip_optimizer_step):
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        
            # visualize the grad norm of each layer of our transformer (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}  # Dictionary to store norms per layer
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Some parameters might not have gradients
                        grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                for layer_name, grad_norm in grad_norms.items():
                    wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                if total_grad_norm > config.training.grad_clip_norm * 2.0:
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                # show grad norm in wandb if it's too large
                display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                if display_grad_norm and ddp_info.is_main_process:
                    wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

            if not skip_optimizer_step:
                scaler.step(optimizer)
                scaler.update()
                cur_param_update_step += 1

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # log and save checkpoint
        if ddp_info.is_main_process:
            loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
            # print in console
            if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
                print_str = f"[Epoch {int(cur_epoch):>3d}] | Total samples: {cur_train_step * total_batch_size} | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
                print_str += f" | Iter Time: {time.time() - tic:.2f}s | Data Time: {data_time:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
                # Add loss values
                for k, v in loss_dict.items():
                    if 'lpips' in k:
                        continue
                    print_str += f"{k}: {v:.6f} | "
                print(print_str)

            # log in wandb
            if (cur_train_step % config.training.wandb_log_every == 0) or (
                cur_train_step < 200 + start_train_step
            ):
                current_time = datetime.now()
                total_time_seconds = (current_time - training_start_time).total_seconds()
                total_time_hours = total_time_seconds / 3600
                
                # 基础日志数据
                base_log_dict = {
                    "iter": cur_train_step,
                    "Data Time": data_time,
                    "lr": optimizer.param_groups[0]["lr"],
                    "iter_time": time.time() - tic,
                    "grad_norm": total_grad_norm,
                    "epoch": cur_epoch,
                    "total_time_seconds": total_time_seconds,
                    "total_time_hours": total_time_hours,
                    "param_update_step": cur_param_update_step,
                    "total_samples": cur_train_step * total_batch_size,
                    "total_views": cur_train_step * total_batch_size * config.training.num_target_views
                }
                
                # 添加损失指标
                loss_log_dict = {"train/" + k: v for k, v in loss_dict.items()}
                base_log_dict.update(loss_log_dict)
                
                # 记录到文件日志
                file_only_logger.info(base_log_dict)
                
                wandb.log(
                    base_log_dict,
                    step=cur_train_step * total_batch_size,
                )
                
                base_log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
                file_only_logger.info(base_log_dict)      

            # save checkpoint
            current_time = datetime.now()
            time_since_last_save = (current_time - last_time_save).total_seconds()
            should_save_by_time = (time_since_last_save >= save_every_seconds) or (current_time - training_start_time).total_seconds() >= total_train_seconds
            should_save_by_steps = (cur_train_step % save_every_steps == 0) or (cur_train_step == total_train_steps)
            
            if should_save_by_steps or should_save_by_time:
                if isinstance(model, DDP):
                    model_weights = model.module.state_dict()
                else:
                    model_weights = model.state_dict()
                checkpoint = {
                    "model": model_weights,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "fwdbwd_pass_step": cur_train_step,
                    "param_update_step": cur_param_update_step,
                }
                os.makedirs(config.training.checkpoint_dir, exist_ok=True)
                
                # 生成检查点文件名
                if should_save_by_steps:
                    # 基于epoch的保存
                    ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_epoch * 100:04f}.pt")
                    save_reason = f"epoch*100 {cur_epoch * 100}"
                else:
                    # 基于时间的保存
                    hours_elapsed = (current_time - training_start_time).total_seconds() / 3600
                    ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_t{hours_elapsed:.3f}h.pt")
                    save_reason = f"time {hours_elapsed:.3f}h"
                
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint at {save_reason} to {os.path.abspath(ckpt_path)}")
                
                # 如果是因为时间保存的，更新上次保存时间
                if should_save_by_time:
                    last_time_save = current_time
            
            # export intermediate visualization results
            if export_inter_results:
                vis_path = os.path.join(config.training.checkpoint_dir, 'visualize')
                vis_path = os.path.join(vis_path, f"iter_{cur_train_step:08d}")
                os.makedirs(vis_path, exist_ok=True)
                visualize_intermediate_results(vis_path, ret_dict)
                torch.cuda.empty_cache()
                model.train()

                
        if export_inter_results:
            torch.cuda.empty_cache()
            dist.barrier()


if __name__ == "__main__":
    main()
