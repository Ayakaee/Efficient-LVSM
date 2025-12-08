# Incremental Inference for LVSM

This guide explains how to use the incremental inference feature in the LVSM project.

## Overview

Incremental inference is a specialized mode that processes input views sequentially, exporting results after each view is processed. This is particularly useful for:

- Analyzing how model performance improves with additional input views
- Memory-constrained scenarios
- Progressive rendering applications
- Performance benchmarking

## Key Features

- **Sequential Processing**: Processes views one by one (1 view, 2 views, 3 views, etc.)
- **Incremental Export**: Saves results after each view is added
- **KV Cache Optimization**: Uses key-value caching for improved efficiency
- **Memory Tracking**: Monitors GPU memory usage at each step
- **Distributed Support**: Works with multi-GPU setups

## Quick Start

### Basic Usage

```bash
python inference_incremental.py --config configs/LVSM_scene_decoder_only.yaml
```

### Multi-GPU Usage

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    inference_incremental.py --config configs/LVSM_scene_decoder_only.yaml
```

## Configuration

Key configuration parameters in your YAML file:

```yaml
training:
  num_input_views: 4          # Number of input views to process
  num_target_views: 1         # Number of target views to render
  batch_size_per_gpu: 1       # Recommended: 1 for incremental inference
  target_has_input: false     # Whether target includes input views
  use_amp: true               # Use automatic mixed precision
  amp_dtype: "bf16"           # Mixed precision dtype

inference:
  checkpoint_dir: "./experiments/evaluation/incremental"  # Output directory
  compute_metrics: true       # Calculate PSNR/SSIM/LPIPS
  render_video: true          # Render interpolated video
  resize: false               # Resize output images
```

## Output Structure

Results are organized in subdirectories:

```
checkpoint_dir/incremental/
├── view_000/          # Results with 1 input view
│   ├── images/        # Rendered and ground truth images
│   ├── metrics.json   # Evaluation metrics
│   └── video/         # Rendered video (if enabled)
├── view_001/          # Results with 2 input views
├── view_002/          # Results with 3 input views
└── view_003/          # Results with 4 input views
```

## How It Works

### Workflow

1. **Initialization**
   - Load model and dataset
   - Clear KV cache
   - Create output directories

2. **Incremental Processing Loop**
   For each input view i (i = 1 to N):
   - Extract view i from input data
   - Perform forward pass with views 1 through i
   - Update KV cache with new view information
   - Render target views
   - Export results to `view_{i-1:03d}/`
   - Track GPU memory usage

3. **Finalization**
   - Compute aggregate metrics
   - Generate summary report

### KV Cache Mechanism

The incremental inference leverages key-value caching to avoid redundant computations:

```python
# First view: Compute and cache KV
model.module.clear_kv_cache()
result = model(..., incremental_mode=True)  # Caches KV for view 1

# Second view: Reuse cached KV, only compute new view
result = model(..., incremental_mode=True)  # Uses cached KV, adds view 2

# Subsequent views follow the same pattern
```

This reduces computation significantly for later views.

## Testing

### Functional Test

Test the incremental inference pipeline without a trained model:

```bash
python test_incremental_inference.py --num_views 4 --save_results
```

### Performance Comparison

Compare incremental vs. normal inference:

```bash
python compare_inference.py --max_views 8 --num_trials 3
```

This generates:
- CSV file with detailed performance metrics
- Plots comparing time and memory usage
- Summary report with speedup ratios

## Advanced Usage

### Custom Number of Views

Process a specific number of input views:

```bash
python inference_incremental.py \
    --config configs/LVSM_scene_decoder_only.yaml \
    training.num_input_views=8 \
    training.num_target_views=1
```

### Memory Profiling

Track memory usage for each view:

```bash
python inference_incremental.py \
    --config configs/LVSM_scene_decoder_only.yaml \
    2>&1 | tee incremental_memory_log.txt
```

### Batch Processing

Process multiple scenes (requires custom script):

```bash
for scene in scene_001 scene_002 scene_003; do
    python inference_incremental.py \
        --config configs/LVSM_scene_decoder_only.yaml \
        training.scene_filter="$scene" \
        inference.checkpoint_dir="./experiments/$scene/incremental"
done
```

## Comparison with Normal Inference

| Feature | Normal Inference | Incremental Inference |
|---------|------------------|----------------------|
| Input Processing | All views at once | One view at a time |
| Result Export | Final result only | After each view |
| KV Caching | Not used | Used for efficiency |
| Memory Usage | Higher (linear growth) | Lower (cached) |
| Speed (few views) | Similar | Slightly slower |
| Speed (many views) | Slower | Significantly faster |
| Use Case | Standard evaluation | Analysis, profiling |

## Troubleshooting

### CUDA Out of Memory

**Solution**:
- Reduce `num_target_views`
- Lower image resolution
- Use smaller batch size
- Enable mixed precision (`use_amp: true`)

### KV Cache Not Working

**Solution**:
- Ensure model has `clear_kv_cache()` method
- Verify `incremental_mode=True` is passed to forward()
- Check that model supports incremental inference mode

### Results Don't Match Normal Inference

**Explanation**: This is expected. Incremental inference processes views sequentially, so intermediate results will differ. The final result (with all views) should match normal inference closely.

## Performance Tips

1. **Batch Size**: Use `batch_size_per_gpu=1` for incremental inference
2. **Mixed Precision**: Enable `use_amp: true` and `amp_dtype: "bf16"`
3. **Video Rendering**: Disable if not needed to save time
4. **Metrics**: Set `compute_metrics: false` for faster inference

## Example Workflow

1. **Initial Setup**
   ```bash
   # Download checkpoint
   wget https://huggingface.co/coast01/LVSM/resolve/main/scene_decoder_only_256.pt
   
   # Prepare data
   python scripts/process_data.py --base_path /path/to/data --mode test
   ```

2. **Run Incremental Inference**
   ```bash
   python inference_incremental.py --config configs/LVSM_scene_decoder_only.yaml
   ```

3. **Analyze Results**
   ```python
   import json
   import glob
   
   # Load metrics from all views
   metrics_files = sorted(glob.glob('experiments/incremental/view_*/metrics.json'))
   for i, f in enumerate(metrics_files):
       with open(f) as fp:
           metrics = json.load(fp)
       print(f"Views: {i+1}, PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.3f}")
   ```

## API Reference

### Main Script

**File**: `inference_incremental.py`

**Key Functions**:
- Loads configuration and model
- Processes views incrementally
- Exports results for each step
- Tracks memory usage

**Command Line Args**:
- `--config`: Path to configuration YAML file
- Additional args override config values (e.g., `training.num_input_views=8`)

### Test Script

**File**: `test_incremental_inference.py`

**Usage**:
```bash
python test_incremental_inference.py [options]
```

**Options**:
- `--num_views INT`: Number of views to test (default: 4)
- `--save_results`: Save test results to disk
- `--test_cache`: Test KV cache logic

### Comparison Script

**File**: `compare_inference.py`

**Usage**:
```bash
python compare_inference.py [options]
```

**Options**:
- `--max_views INT`: Maximum number of views (default: 8)
- `--num_trials INT`: Number of test runs (default: 3)
- `--image_size INT`: Image resolution (default: 256)
- `--output_dir STR`: Output directory (default: 'test_output')
- `--no_plot`: Skip plot generation

## Citation

If you use incremental inference in your research, please cite the LVSM paper:

```bibtex
@inproceedings{
jin2025lvsm,
title={LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias},
author={Haian Jin and Hanwen Jiang and Hao Tan and Kai Zhang and Sai Bi and Tianyuan Zhang and Fujun Luan and Noah Snavely and Zexiang Xu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=QQBPWtvtcn}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project follows the same license as the main LVSM project.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the main README for general LVSM questions
- Contact the maintainers

## Acknowledgments

This incremental inference implementation builds upon the LVSM codebase. Special thanks to the LVSM authors and contributors.

