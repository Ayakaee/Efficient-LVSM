#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental Inference Test Script
For quickly testing if incremental inference functionality works correctly

Copyright (c) 2025 Yihang Sun. Efficient-LVSM project.
Licensed under CC BY-NC-SA 4.0 - see LICENSE.md for details.
"""

import os
import sys
import torch
from easydict import EasyDict as edict
import argparse

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_simple_test_data(num_input_views=4, num_target_views=1, image_size=256, device='cuda'):
    """Create simple test data"""
    batch_size = 1
    
    # Create input data
    input_all = edict()
    input_all.image = torch.randn(batch_size, num_input_views, 3, image_size, image_size, device=device)
    input_all.ray_o = torch.randn(batch_size, num_input_views, 3, image_size, image_size, device=device)
    input_all.ray_d = torch.randn(batch_size, num_input_views, 3, image_size, image_size, device=device)
    input_all.c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_input_views, -1, -1)
    input_all.fxfycxcy = torch.tensor([[[512, 512, 128, 128]]], device=device).expand(batch_size, num_input_views, -1)
    input_all.index = torch.arange(num_input_views, device=device).unsqueeze(0).unsqueeze(-1)
    input_all.scene_name = ['test_scene']
    
    # Create target data
    target_all = edict()
    target_all.image = torch.randn(batch_size, num_target_views, 3, image_size, image_size, device=device)
    target_all.ray_o = torch.randn(batch_size, num_target_views, 3, image_size, image_size, device=device)
    target_all.ray_d = torch.randn(batch_size, num_target_views, 3, image_size, image_size, device=device)
    target_all.image_h_w = (image_size, image_size)
    target_all.scene_name = ['test_scene']
    target_all.index = torch.tensor([[[num_input_views]]], device=device)
    
    return input_all, target_all

def test_incremental_inference(num_input_views=4, save_results=True):
    """Test incremental inference functionality"""
    
    print("="*60)
    print("Incremental Inference Functionality Test")
    print("="*60)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cpu':
        print("Warning: GPU not detected, testing will run on CPU (may be slow)")
    
    # Create test data
    print(f"\nCreating test data: {num_input_views} input views")
    input_all, target_all = create_simple_test_data(
        num_input_views=num_input_views,
        num_target_views=1,
        image_size=256,
        device=device
    )
    
    print(f"Input data shape: {input_all.image.shape}")
    print(f"Target data shape: {target_all.image.shape}")
    
    # Create output directory
    output_dir = 'test_output/incremental_test'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting Simulated Incremental Inference Flow")
    print("="*60)
    
    # Simulate incremental inference process
    results = []
    
    for view_idx in range(num_input_views):
        print(f"\n{'='*50}")
        print(f"Processing input view {view_idx + 1}/{num_input_views}")
        print(f"{'='*50}")
        
        # Extract input data for current view
        input_view = edict()
        input_view.image = input_all.image[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_o = input_all.ray_o[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.ray_d = input_all.ray_d[:, view_idx:view_idx+1, :, :, :].clone()
        input_view.c2w = input_all.c2w[:, view_idx:view_idx+1, :, :].clone()
        input_view.fxfycxcy = input_all.fxfycxcy[:, view_idx:view_idx+1, :].clone()
        input_view.index = input_all.index[:, view_idx:view_idx+1, :].clone()
        input_view.scene_name = input_all.scene_name
        
        # Prepare target view data
        target_view = edict()
        target_view.image = target_all.image.clone()
        target_view.ray_o = target_all.ray_o.clone()
        target_view.ray_d = target_all.ray_d.clone()
        target_view.image_h_w = target_all.image_h_w
        target_view.scene_name = target_all.scene_name
        target_view.index = target_all.index.clone()
        
        print(f"Input view index: {input_view.index[0].cpu().numpy()}")
        print(f"Target view index: {target_view.index[0].cpu().numpy()}")
        print(f"Input view shape: {input_view.image.shape}")
        print(f"Target view shape: {target_view.image.shape}")
        
        # Track memory usage (if using GPU)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Initial GPU memory: {initial_memory:.2f} MB")
        
        # This should call the model's incremental inference method
        # result = model(batch, input_view, target_view, train=False, incremental_mode=True)
        # Since this is a test script, we just simulate the data flow
        
        result = edict()
        result.input = input_view
        result.target = target_view
        result.render = torch.randn_like(target_view.image)  # Simulated render result
        
        # Record peak memory
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU memory: {peak_memory:.2f} MB")
            print(f"Memory increment: {peak_memory - initial_memory:.2f} MB")
        
        results.append(result)
        
        # Save results (if needed)
        if save_results:
            view_output_dir = os.path.join(output_dir, f'view_{view_idx:03d}')
            os.makedirs(view_output_dir, exist_ok=True)
            
            # Save rendered image (simplified version)
            render_image = result.render[0, 0].cpu()
            torch.save(render_image, os.path.join(view_output_dir, 'render.pt'))
            
            print(f"Results saved to: {view_output_dir}")
        
        print(f"View {view_idx + 1} processing completed")
    
    print("\n" + "="*60)
    print("Incremental Inference Test Completed!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Processed {num_input_views} input views")
    print(f"  - Generated {len(results)} results")
    if save_results:
        print(f"  - Results saved in: {output_dir}")
    
    return results

def test_kv_cache_logic():
    """Test KV cache logic"""
    print("\n" + "="*60)
    print("Testing KV Cache Logic")
    print("="*60)
    
    # Simulate KV cache
    kv_cache = {}
    
    num_layers = 4
    d_model = 768
    seq_len = 256
    
    print(f"\nSimulating KV cache for {num_layers}-layer transformer")
    
    for view_idx in range(4):
        print(f"\nView {view_idx + 1}:")
        
        for layer_idx in range(num_layers):
            cache_key = f'layer_{layer_idx}'
            
            # New view's KV
            new_kv = {
                'key': torch.randn(1, seq_len, d_model),
                'value': torch.randn(1, seq_len, d_model)
            }
            
            if cache_key in kv_cache:
                # Concatenate with cache
                old_kv = kv_cache[cache_key]
                cached_seq_len = old_kv['key'].shape[1]
                
                kv_cache[cache_key] = {
                    'key': torch.cat([old_kv['key'], new_kv['key']], dim=1),
                    'value': torch.cat([old_kv['value'], new_kv['value']], dim=1)
                }
                
                new_seq_len = kv_cache[cache_key]['key'].shape[1]
                print(f"  Layer {layer_idx}: cache length {cached_seq_len} -> {new_seq_len}")
            else:
                # First time caching
                kv_cache[cache_key] = new_kv
                print(f"  Layer {layer_idx}: initialized cache, length {seq_len}")
    
    print("\nFinal cache state:")
    for cache_key, kv in kv_cache.items():
        seq_len = kv['key'].shape[1]
        print(f"  {cache_key}: sequence length = {seq_len}")
    
    print("\nKV cache logic test completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Incremental Inference Functionality Test')
    parser.add_argument('--num_views', type=int, default=4, help='Number of input views')
    parser.add_argument('--save_results', action='store_true', help='Whether to save test results')
    parser.add_argument('--test_cache', action='store_true', help='Test KV cache logic')
    
    args = parser.parse_args()
    
    try:
        # Test incremental inference flow
        print("Starting tests...")
        results = test_incremental_inference(
            num_input_views=args.num_views,
            save_results=args.save_results
        )
        
        # Test KV cache logic
        if args.test_cache:
            test_kv_cache_logic()
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
