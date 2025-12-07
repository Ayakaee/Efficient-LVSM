#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
普通模型测试程序
测试普通推理模型的性能
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import argparse

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_config():
    """Create test configuration"""
    config = edict()
    
    # 模型配置
    config.model = edict()
    config.model.concat_rgb = True
    config.model.image_tokenizer = edict()
    config.model.image_tokenizer.type = 'none'  # 简化测试，不使用图像编码器
    config.model.image_tokenizer.in_channels = 9  # 3 + 6
    config.model.image_tokenizer.patch_size = 8
    config.model.image_tokenizer.image_size = 256
    config.model.target_pose_tokenizer = edict()
    config.model.target_pose_tokenizer.in_channels = 6
    config.model.target_pose_tokenizer.patch_size = 8
    config.model.transformer = edict()
    config.model.transformer.d = 768
    config.model.transformer.d_head = 24
    config.model.transformer.n_layer = 24
    config.model.transformer.mode = 'alternate'
    config.model.transformer.input_mode = 'embed'
    config.model.transformer.attention_arch = 'flex'
    config.model.transformer.use_qk_norm = True
    config.model.transformer.use_log_scale = None
    config.model.transformer.encoder_n_layer = 6
    config.model.transformer.decoder_n_layer = 18
    config.model.transformer.n_latent_vectors = 3072 # 3x32x32
    
    # 训练配置
    config.training = edict()
    config.training.enable_repa = False
    config.training.use_amp = True
    config.training.amp_dtype = 'bf16'
    config.training.target_has_input = False
    config.training.grad_checkpoint_every = 1
    config.training.lpips_loss_weight = 0.0
    config.training.perceptual_loss_weight = 0.5
    config.training.proj_loss_weight = 0.5
    config.training.l2_loss_weight = 0.5
    
    # 禁用分布式训练相关功能
    config.training.use_ddp = False
    config.training.distributed = False
    
    # 禁用增量推理
    config.enable_incremental_inference = False
    
    return config

def create_dummy_data(batch_size=1, num_views=1, image_size=256, device='cpu'):
    """Create dummy test data"""
    # 创建虚拟输入数据
    input_data = edict()
    input_data.image = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    input_data.ray_o = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    input_data.ray_d = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    input_data.c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1)
    input_data.fxfycxcy = torch.tensor([[[512, 512, 128, 128]]], device=device).expand(batch_size, num_views, -1)
    
    # 创建虚拟目标数据
    target_data = edict()
    v_target = 1
    target_data.image = torch.randn(batch_size, v_target, 3, image_size, image_size, device=device)
    target_data.ray_o = torch.randn(batch_size, v_target, 3, image_size, image_size, device=device)
    target_data.ray_d = torch.randn(batch_size, v_target, 3, image_size, image_size, device=device)
    target_data.image_h_w = (image_size, image_size)
    
    return input_data, target_data

class MockLogger:
    """Mock logger for testing"""
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")

def test_normal_model():
    """测试普通推理模型性能"""
    
    # 设置设备
    # from setup import init_config, init_distributed, init_logging
    # ddp_info = init_distributed(seed=777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置和模型
    config = create_test_config()
    from model.incremental2 import Images2LatentScene
    logger = MockLogger()
    model = Images2LatentScene(config, logger)
    model = model.to(device)
    model.eval()
    
    # 测试参数
    max_views = 16
    batch_size = 1
    image_size = 256
    num_warmup = 0  # 预热次数
    num_trials = 1  # 每个测试的重复次数
    
    print(f"开始普通推理模型测试：从1个视图到{max_views}个视图")
    print(f"预热次数: {num_warmup}, 测试次数: {num_trials}")
    
    # 存储结果
    results = []
    
    # 初始化结果列表，用于累积所有trial的结果
    normal_times = [[] for _ in range(max_views)]
    normal_memory = [[] for _ in range(max_views)]
    
    with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=torch.bfloat16,
):
        # 为每个视图数量进行测试
        for trial in range(num_trials + num_warmup):
            print(f"\n第 {trial + 1} 次运行...")
            
            # 进行多次测试以取平均
            for num_views in range(1, max_views + 1):
                # 创建测试数据
                input_data, target_data = create_dummy_data(
                    batch_size=batch_size, 
                    num_views=num_views, 
                    image_size=image_size, 
                    device=device
                )
                
                # 测试普通推理
                # 记录内存使用
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                
                # 普通推理：一次性处理所有视图
                start_time = time.time()
                
                with torch.no_grad():
                    result = model.forward(
                        data_batch=None,
                        input=input_data,
                        target=target_data,
                        has_target_image=False,
                        use_incremental=False,
                        train=False
                    )
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                print(end_time - start_time)
                
                # 记录内存峰值
                if device.type == 'cuda':
                    memory_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
                
                if trial >= num_warmup:
                    normal_times[num_views - 1].append(end_time - start_time)
                    if device.type == 'cuda':
                        normal_memory[num_views - 1].append(memory_peak)
                
                # 清理内存
                del input_data, target_data, result
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
    # 计算每个视图数的平均时间和内存使用
    for num_views in range(1, max_views + 1):
        idx = num_views - 1
        avg_normal_time = np.mean(normal_times[idx]) if normal_times[idx] else 0
        avg_normal_memory = np.mean(normal_memory[idx]) if normal_memory[idx] else 0
        # 记录结果
        results.append({
            'model_type': 'normal',
            'num_views': num_views,
            'avg_time': avg_normal_time,
            'avg_memory': avg_normal_memory,
            'num_trials': num_trials
        })

    return results

def save_results_to_csv(results, filename='test_data/normal_model_results.csv'):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"普通推理模型测试结果已保存到: {filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='普通推理模型测试')
    parser.add_argument('--max_views', type=int, default=16, help='最大视图数量')
    parser.add_argument('--output', type=str, default='test_data/inc-ours.csv', help='输出CSV文件名')
    
    args = parser.parse_args()
    
    print("开始普通推理模型测试...")
    print(f"最大视图数: {args.max_views}")
    
    try:
        # 运行测试
        results = test_normal_model()
        
        # 保存结果
        save_results_to_csv(results, args.output)
        
        # 打印总结
        print("\n测试完成！")
        print("结果总结:")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
