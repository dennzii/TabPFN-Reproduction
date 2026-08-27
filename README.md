# TabPFN Reproduction

A clean PyTorch reproduction of [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) (ICLR 2023), implementing the synthetic Structural Causal Model (SCM) prior data generation pipeline and Transformer architecture from scratch.

---

## 📌 Implemented SCM Prior Parameters

The prior parameters highlighted in yellow from **Table 5** of the TabPFN paper are implemented in this repository:

<img width="819" height="442" alt="image" src="https://github.com/user-attachments/assets/ee07e237-dc9c-4a9c-9f14-e6958893bb53" />
*From TabPFN Paper (Table 5)*

* **MLP Weight Dropout:** $0.9 \cdot \text{Beta}(a, b)$, with $a, b \sim \text{Uniform}(0.1, 5.0)$
* **Target Sampling:** Option to sample $y$ directly from the last MLP layer (`True/False`)
* **Activation Functions:** Uniform sampling across `{Tanh, Leaky ReLU, ELU, Identity}`
* **Blockwise Dropout & Feature Ordering:** Preserves causal feature groups and block structure
* **Layer & Node Distributions (TNLU):**
  * MLP Number of Layers: 2 – 6
  * Hidden Nodes per Layer: 4 – 130
  * Gaussian Noise Std: 0.0001 – 0.3
  * MLP Weights Std: 0.01 – 10.0
  * Layer 1 SCM Nodes: 1 – 12

---

## ⚙️ Training Setup

* **Hardware:** NVIDIA G4 / RTX 6000 Ada (96 GB VRAM)
* **Training Time:** ~10 hours
* **Total Steps:** 18,000 optimization steps
* **Batch Size:** 256 (256 synthetic datasets per gradient step)
* **Max Sequence Length ($N$):** 1024 samples
* **Max Features ($K$):** 100 features (Zero-padded & scaled by $K / k$)
* **Max Classes ($C$):** Up to 10 classes
* **Optimization & Learning Rate Schedule:**
  * Optimizer: Adam
  * Peak LR: `1e-4`
  * Warmup: 1,000 steps (Linear)
  * Decay: Cosine Annealing (`1e-4` $\to$ `0.0`)
  * Loss: Cross-Entropy Loss

---

## 📉 Training Loss Curve

Exponential Moving Average (EMA) cross-entropy loss recorded every 100 steps during pre-training:

<img width="1000" height="600" alt="tabpfn_loss_curve18000" src="https://github.com/user-attachments/assets/588982ba-1f8e-47c5-a6da-97556c841649" />

---

## Pre-trained Checkpoint

Pre-trained weights at step 18,000 are available for download:
* **Google Drive:** [tabpfn_step_18000.pt (Download)](https://drive.google.com/file/d/19S0lVQJDsNzIcI4pxF6qIXM8-mag-1G3/view?usp=sharing)

---

## Benchmark Results (30 OpenML-CC18 Datasets)

Evaluated on the official meta-test suite of **30 OpenML-CC18** datasets (50/50 Train-Test split) following the paper's evaluation protocol:

| Variant | Mean ROC AUC (OVO) | Mean Accuracy | Mean Log Loss |
| :--- | :---: | :---: | :---: |
| **Single Pass (No Ensemble - `n=1`)** | 0.8668 | 79.27% | 0.4918 |
| **32-Permutation Ensemble (`n=32`)** | **0.8706** | **80.12%** | **0.4839** |
| *Original TabPFN n.e. (Paper Table 2)* | *0.8910* | *82.00%* | *0.7420* |

> For the full dataset-by-dataset comparison and breakdown, see [tabpfn_comparison_table.png](tabpfn_comparison_table.png).

---

## Quickstart & Inference

### 1. Installation
```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy tqdm matplotlib



```

## References
@inproceedings{hollmann2023tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Noah Hollmann and Samuel M{\"u}ller and Katharina Eggensperger and Frank Hutter},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
