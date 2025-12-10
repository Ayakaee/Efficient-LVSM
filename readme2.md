<div align="center">

# Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model <br> via Decoupled Co-Refinement Attention

<a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b.svg"></a>
<a href="https://huggingface.co/YourUsername/Efficient-LVSM"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow"></a>
<a href="https://your-project-page.github.io"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
<a href="https://github.com/YourGithub/Efficient-LVSM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>

**Xiaosong Jia**<sup>1,2,*</sup>, **Yihang Sun**<sup>2,*</sup>, **Junqi You**<sup>2</sup>, **Songbur Wong**<sup>2</sup>, **Zichen Zou**<sup>2</sup>, **Junchi Yan**<sup>2</sup>, **Zuxuan Wu**<sup>1</sup>, **Yu-Gang Jiang**<sup>1</sup>

<sup>1</sup>Institute of Trustworthy Embodied AI (TEAI), Fudan University <br>
<sup>2</sup>Sch. of Computer Science & Sch. of Artificial Intelligence, Shanghai Jiao Tong University

</div>

---

## 📸 Demo

<div align="center">
  <!-- 这里放你最好的展示GIF，建议展示 RealEstate10K 的漫游效果 -->
  <img src="assets/teaser_demo.gif" width="800px">
  <br>
  <em>Efficient-LVSM enables high-quality novel view synthesis with 4.4x faster inference speed and supports incremental inference via KV-Cache.</em>
</div>

## 📰 News
*   **[202X-XX-XX]** Code and pretrained models are released.
*   **[202X-XX-XX]** Paper is accepted by **ICLR 2026**!
*   **[202X-XX-XX]** Paper uploaded to arXiv.

## 💡 Abstract & Highlights

We propose **Efficient-LVSM**, a dual-stream architecture that decouples input encoding from target decoding. Unlike previous monolithic methods (e.g., LVSM), our approach avoids quadratic complexity and achieves state-of-the-art performance.

*   🚀 **High Efficiency**: **4.4x faster** inference and **2x faster** training convergence compared to LVSM.
*   🧠 **Decoupled Architecture**: Input Encoder (Intra-view Self-Attention) + Target Decoder (Self-then-Cross Attention).
*   💾 **KV-Cache Support**: Enables **incremental inference** (constant cost for adding new views).
*   🏆 **SOTA Performance**: **30.6 dB PSNR** on RealEstate10K (surpassing LVSM by 0.9 dB).
*   🌐 **Strong Generalization**: Zero-shot generalization to unseen numbers of input views.

## 📊 Results

### Quantitative Comparison
Our model outperforms existing state-of-the-art methods on both scene-level (RealEstate10K) and object-level (GSO/ABO) benchmarks while being significantly faster.

| Model | Parameters | Latency (ms) | GFLOPS | PSNR (RealEstate10K) |
| :--- | :---: | :---: | :---: | :---: |
| pixelSplat | 125M | 50.52 | 1934 | 26.09 |
| GS-LRM | 307M | 88.24 | 5047 | 28.10 |
| LVSM (Dec-Only) | 177M | 109.37 | 8523 | 29.67 |
| **Efficient-LVSM (Ours)** | **199M** | **24.78** | **1325** | **30.61** |

### Visual Comparison
<div align="center">
  <!-- 这里放论文里的 Figure 4 或 Figure 6，对比其他方法的截图 -->
  <img src="assets/visual_comparison.png" width="100%">
  <br>
  <em>Comparison with LVSM on RealEstate10K. Our model preserves sharper details and geometry.</em>
</div>

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Efficient-LVSM.git
cd Efficient-LVSM

# 2. Create environment
conda create -n efficient-lvsm python=3.9
conda activate efficient-lvsm

# 3. Install dependencies
# We recommend installing pytorch according to your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt