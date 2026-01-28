# Incremental Inference Benchmark

Unlike standard multi-view inference that assumes all input views are available upfront, this benchmark evaluates **how model performance, efficiency, and memory usage evolve as input views arrive sequentially**.

The benchmark is designed to reflect real-world, online, and resource-constrained deployment scenarios for large-scale view synthesis models.

---

## Motivation

Most existing Novel View Synthesis (NVS) benchmarks assume a *static batch inference* setting:
- All input views are known beforehand
- The model processes them jointly
- Only the final reconstruction quality is evaluated

However, many practical applications operate under **incremental observation** constraints:
- Views arrive over time
- Intermediate predictions are required
- Memory and latency budgets matter as much as final quality

This benchmark aims to close this gap by evaluating **incremental inference behavior**.

---

## What Is Incremental Inference?

In this benchmark, the model:
1. Receives input views **one by one**
2. Updates its internal state (e.g. KV cache, scene representation)
3. Performs inference after each new view
4. Exports intermediate results and metrics at every step

Formally, given a sequence of input views  
$$
\{v_1, v_2, \dots, v_N\},
$$

the model produces predictions
$$
\{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N\},
$$
where $\hat{y}_k$ is inferred using only $\{v_1, \dots, v_k\}$.

---

## Benchmark Usage

For each incremental step, the benchmark records:

1. Reconstruction Quality：PSNR / SSIM / LPIPS

2. Efficiency Metrics
    - Per-step inference latency
    - Peak GPU memory usage
    - Incremental cost vs. full recomputation

**Usage Instructions:**
```
cd increamental_inference_benchmark
bash run_benchmark.sh
```

**Output Structure:**
```
experiments/evaluation/incremental/
├── view_000/  # Results with 1 input view
├── view_001/  # Results with 2 input views
├── view_002/  # Results with 3 input views
└── view_003/  # Results with 4 input views
```


---

## Why This Matters

A model that performs well under full-view inference may still:
- Waste computation by reprocessing previous views
- Fail to scale under memory constraints
- Produce unstable intermediate predictions

The Incremental Inference Benchmark exposes these limitations and enables:
- Fair comparison of incremental vs. non-incremental designs
- Ablation of KV-cache and state reuse strategies
- Analysis of quality–efficiency trade-offs

---

## Notes

This benchmark is **model-agnostic** and can be adapted to other NVS architectures. The current implementation builds upon Efficient-LVSM with explicit KV-cache reuse.  
