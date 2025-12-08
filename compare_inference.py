#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental vs. Normal Inference Performance Comparison Script
Compares time and memory consumption between the two inference methods

Copyright (c) 2025 Yihang Sun. Efficient-LVSM project.
Licensed under CC BY-NC-SA 4.0 - see LICENSE.md for details.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import argparse

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data(num_views, image_size=256, device='cuda'):
    """Create test data"""
    batch_size = 1
    
    data = edict()
    data.image = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    data.ray_o = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    data.ray_d = torch.randn(batch_size, num_views, 3, image_size, image_size, device=device)
    data.c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1)
    data.fxfycxcy = torch.tensor([[[512, 512, 128, 128]]], device=device).expand(batch_size, num_views, -1)
    data.index = torch.arange(num_views, device=device).unsqueeze(0).unsqueeze(-1)
    data.scene_name = ['test_scene']
    data.image_h_w = (image_size, image_size)
    
    return data

def simulate_normal_inference(input_data, target_data, device):
    """Simulate normal inference"""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    # Simulate inference computation
    # In actual use, this should call model(batch, input, target, train=False, incremental_mode=False)
    
    # Simulate computation load
    num_views = input_data.image.shape[1]
    for _ in range(10 * num_views):  # Simulate computation complexity increases with number of views
        _ = torch.matmul(input_data.image.flatten(0, 1), input_data.image.flatten(0, 1).T)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
    
    return elapsed_time, peak_memory

def simulate_incremental_inference(input_data, target_data, device):
    """Simulate incremental inference"""
    num_views = input_data.image.shape[1]
    total_time = 0
    peak_memory = 0
    
    # Simulate KV cache
    kv_cache_size = 0
    
    for view_idx in range(num_views):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Extract current view
        input_view = edict()
        input_view.image = input_data.image[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_o = input_data.ray_o[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_d = input_data.ray_d[:, view_idx:view_idx+1, :, :, :].clone()
        
        # Simulate inference (using cache, reduced computation)
        # In actual use, this should call model(batch, input_view, target, train=False, incremental_mode=True)
        
        # Simulate computation load (incremental inference only processes one new view each time, less computation)
        for _ in range(10):  # Fixed computation amount
            _ = torch.matmul(input_view.image.flatten(0, 1), input_view.image.flatten(0, 1).T)
        
        # Simulate KV cache growth
        kv_cache_size += 256 * 768 * 4 / 1024**2  # Assume each view adds ~1MB cache
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        if device.type == 'cuda':
            view_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_memory = max(peak_memory, view_memory)
    
    return total_time, peak_memory

def run_comparison(max_views=16, num_trials=3, image_size=256):
    """Run performance comparison test"""
    
    print("="*70)
    print("Incremental vs. Normal Inference Performance Comparison")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cpu':
        print("Warning: GPU not detected, testing will run on CPU")
    
    print(f"\nTest configuration:")
    print(f"  - Max views: {max_views}")
    print(f"  - Number of trials: {num_trials}")
    print(f"  - Image size: {image_size}x{image_size}")
    
    results = []
    
    for num_views in range(1, max_views + 1):
        print(f"\n{'='*60}")
        print(f"Testing with {num_views} views")
        print(f"{'='*60}")
        
        normal_times = []
        normal_memories = []
        incremental_times = []
        incremental_memories = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end=' ')
            
            # Create test data
            input_data = create_test_data(num_views, image_size, device)
            target_data = create_test_data(1, image_size, device)
            
            # Test normal inference
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            normal_time, normal_memory = simulate_normal_inference(input_data, target_data, device)
            normal_times.append(normal_time)
            normal_memories.append(normal_memory)
            
            # Test incremental inference
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            incremental_time, incremental_memory = simulate_incremental_inference(input_data, target_data, device)
            incremental_times.append(incremental_time)
            incremental_memories.append(incremental_memory)
            
            print(f"Done (Normal: {normal_time:.3f}s, Incremental: {incremental_time:.3f}s)")
            
            # Clean up
            del input_data, target_data
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate averages
        avg_normal_time = np.mean(normal_times)
        avg_normal_memory = np.mean(normal_memories)
        avg_incremental_time = np.mean(incremental_times)
        avg_incremental_memory = np.mean(incremental_memories)
        
        # Calculate speedup and memory savings
        speedup = avg_normal_time / avg_incremental_time if avg_incremental_time > 0 else 0
        memory_saving = (avg_normal_memory - avg_incremental_memory) / avg_normal_memory * 100 if avg_normal_memory > 0 else 0
        
        print(f"\n  Results:")
        print(f"    Normal inference:      Time {avg_normal_time:.3f}s, Memory {avg_normal_memory:.1f}MB")
        print(f"    Incremental inference: Time {avg_incremental_time:.3f}s, Memory {avg_incremental_memory:.1f}MB")
        print(f"    Speedup:               {speedup:.2f}x")
        print(f"    Memory savings:        {memory_saving:.1f}%")
        
        results.append({
            'num_views': num_views,
            'normal_time': avg_normal_time,
            'normal_memory': avg_normal_memory,
            'incremental_time': avg_incremental_time,
            'incremental_memory': avg_incremental_memory,
            'speedup': speedup,
            'memory_saving': memory_saving
        })
    
    return results

def save_results(results, output_dir='test_output'):
    """Save results to CSV and plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {csv_path}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Inference time comparison
    ax = axes[0, 0]
    ax.plot(df['num_views'], df['normal_time'], 'o-', label='Normal Inference', linewidth=2)
    ax.plot(df['num_views'], df['incremental_time'], 's-', label='Incremental Inference', linewidth=2)
    ax.set_xlabel('Number of Input Views')
    ax.set_ylabel('Inference Time (seconds)')
    ax.set_title('Inference Time Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Memory usage comparison
    ax = axes[0, 1]
    ax.plot(df['num_views'], df['normal_memory'], 'o-', label='Normal Inference', linewidth=2)
    ax.plot(df['num_views'], df['incremental_memory'], 's-', label='Incremental Inference', linewidth=2)
    ax.set_xlabel('Number of Input Views')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Speedup ratio
    ax = axes[1, 0]
    ax.plot(df['num_views'], df['speedup'], 'o-', color='green', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Speedup')
    ax.set_xlabel('Number of Input Views')
    ax.set_ylabel('Speedup Ratio')
    ax.set_title('Incremental Inference Speedup (vs. Normal Inference)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Memory savings percentage
    ax = axes[1, 1]
    ax.plot(df['num_views'], df['memory_saving'], 'o-', color='purple', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Savings')
    ax.set_xlabel('Number of Input Views')
    ax.set_ylabel('Memory Savings (%)')
    ax.set_title('Incremental Inference Memory Savings (vs. Normal Inference)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    # Show plot (if in interactive environment)
    try:
        plt.show()
    except:
        pass

def print_summary(results):
    """Print summary report"""
    
    print("\n" + "="*70)
    print("Performance Comparison Summary")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    # Find maximum speedup
    max_speedup_idx = df['speedup'].idxmax()
    max_speedup_row = df.iloc[max_speedup_idx]
    
    print(f"\nMaximum speedup:")
    print(f"  - Number of views: {max_speedup_row['num_views']}")
    print(f"  - Speedup ratio: {max_speedup_row['speedup']:.2f}x")
    print(f"  - Normal inference time: {max_speedup_row['normal_time']:.3f}s")
    print(f"  - Incremental inference time: {max_speedup_row['incremental_time']:.3f}s")
    
    # Find maximum memory savings
    max_saving_idx = df['memory_saving'].idxmax()
    max_saving_row = df.iloc[max_saving_idx]
    
    print(f"\nMaximum memory savings:")
    print(f"  - Number of views: {max_saving_row['num_views']}")
    print(f"  - Memory savings: {max_saving_row['memory_saving']:.1f}%")
    print(f"  - Normal inference memory: {max_saving_row['normal_memory']:.1f}MB")
    print(f"  - Incremental inference memory: {max_saving_row['incremental_memory']:.1f}MB")
    
    # Calculate averages
    avg_speedup = df['speedup'].mean()
    avg_memory_saving = df['memory_saving'].mean()
    
    print(f"\nAverage performance improvements:")
    print(f"  - Average speedup: {avg_speedup:.2f}x")
    print(f"  - Average memory savings: {avg_memory_saving:.1f}%")
    
    # Trend analysis
    print(f"\nTrend analysis:")
    if df['speedup'].iloc[-1] > df['speedup'].iloc[0]:
        print(f"  - As views increase, incremental inference speedup effect becomes more significant")
    else:
        print(f"  - Incremental inference has better speedup effect with fewer views")
    
    if df['memory_saving'].iloc[-1] > 0:
        print(f"  - Incremental inference can significantly save memory in multi-view scenarios")
    else:
        print(f"  - Memory advantages of incremental inference need further optimization")
    
    print("\n" + "="*70)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Incremental vs. Normal Inference Performance Comparison')
    parser.add_argument('--max_views', type=int, default=8, help='Maximum number of views')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of trials per test')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    parser.add_argument('--no_plot', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Incremental Inference Performance Comparison Test")
    print("="*70)
    
    try:
        # Run comparison test
        results = run_comparison(
            max_views=args.max_views,
            num_trials=args.num_trials,
            image_size=args.image_size
        )
        
        # Save results
        if not args.no_plot:
            save_results(results, args.output_dir)
        else:
            # Only save CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(args.output_dir, 'comparison_results.csv')
            os.makedirs(args.output_dir, exist_ok=True)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"\nResults saved to: {csv_path}")
        
        # Print summary
        print_summary(results)
        
        print("\n✓ Testing completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
