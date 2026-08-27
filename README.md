# TabPFN Reproduction

A clean PyTorch reproduction of [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) (ICLR 2023), implementing the synthetic Structural Causal Model (SCM) prior data generation pipeline and Transformer architecture from scratch.

Note: Gemini used for theoritical guidance and code reviewing. Entire implementations (except benchmark.py) is done by me.
---

## Implemented SCM Prior Parameters

The prior parameters highlighted in yellow from **Table 5** of the TabPFN paper are implemented in this repository:

<img width="819" height="442" alt="image" src="https://github.com/user-attachments/assets/ee07e237-dc9c-4a9c-9f14-e6958893bb53" />
*From TabPFN Paper (Table 5)*

* **MLP Weight Dropout:** $0.9 \cdot \text{Beta}(a, b)$, with $a, b \sim \text{Uniform}(0.1, 5.0)$
* **Target Sampling:** Option to sample $y$ directly from the last MLP layer (`True/False`)
* **Activation Functions:** Uniform choice across `{Tanh, Leaky ReLU, ELU, Identity}`
* **Blockwise Dropout & Feature Ordering:** Preserves causal feature groups and structural blocks
* **Two-Stage TNLU Sampling ($\text{TNLU}(h \mid \check{\mu}, \hat{\mu}, \min, \text{round})$):**
  * Means and standard deviations are first drawn from a log-uniform distribution: $\mu, \sigma \sim \text{LogUniform}(\check{\mu}, \hat{\mu})$
  * Final quantities are sampled from the resulting truncated normal: $v \sim \text{TruncNormal}(\mu, \sigma^2, a=0, b=\infty)$, rounded, and shifted by $\min$:
    * **MLP #layers:** $\hat{\mu}=6, \check{\mu}=1, \min=2$
    * **MLP #hidden nodes per layer:** $\hat{\mu}=130, \check{\mu}=5, \min=4$
    * **Gaussian Noise Std:** $\hat{\mu}=0.3, \check{\mu}=0.0001, \min=0.0$
    * **MLP Weights Std:** $\hat{\mu}=10.0, \check{\mu}=0.01, \min=0.0$
    * **SCM #nodes at layer 1:** $\hat{\mu}=12, \check{\mu}=1, \min=1$

To be implemented:
* **Sample SCM vs BNN**: Uniform Choice {True, False}
* **Share Noise mean for**: nodes Uniform Choice {True, False}
* **Input feature scaling enabled**: Uniform Choice {True, False}

---

## ⚙️ Training Setup

* **Hardware:** NVIDIA G4 / RTX 6000 Ada (96 GB VRAM) on COLAB
* **Training Time:** ~10 hours
* **Total Steps:** 18,000 optimization steps
* **Batch Size:** 256 (256 synthetic SCM datasets per gradient update)
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

Evaluated on the official meta-test suite of **30 OpenML-CC18** datasets (50/50 Train-Test split). Comparison between the Single Forward Pass (`n_ensemble=1`) and the 32-Permutation Ensemble (`n_ensemble=32`):

| # | Dataset (OpenML) | ROC AUC (n.e.) | ROC AUC (Ens-32) | Δ AUC | Acc % (n.e.) | Acc % (Ens-32) | Δ Acc % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | balance-scale | 0.9886 | 0.9921 | +0.0035 | 95.85% | 95.85% | +0.00% |
| 2 | mfeat-fourier | 0.9648 | 0.9710 | +0.0062 | 75.20% | 78.80% | +3.60% |
| 3 | breast-w | 0.9922 | 0.9920 | -0.0002 | 96.00% | 96.00% | +0.00% |
| 4 | mfeat-karhunen | 0.9851 | 0.9942 | +0.0091 | 87.20% | 94.40% | +7.20% |
| 5 | mfeat-morphological | 0.9560 | 0.9611 | +0.0051 | 70.80% | 73.60% | +2.80% |
| 6 | mfeat-zernike | 0.9801 | 0.9831 | +0.0030 | 80.80% | 82.80% | +2.00% |
| 7 | cmc | 0.7249 | 0.7261 | +0.0012 | 53.80% | 55.00% | +1.20% |
| 8 | credit-approval | 0.8920 | 0.8922 | +0.0002 | 82.03% | 82.32% | +0.29% |
| 9 | credit-g | 0.7596 | 0.7630 | +0.0034 | 72.00% | 71.20% | -0.80% |
| 10 | diabetes | 0.8357 | 0.8326 | -0.0031 | 75.00% | 75.00% | +0.00% |
| 11 | tic-tac-toe | 0.6369 | 0.6408 | +0.0039 | 65.34% | 65.34% | +0.00% |
| 12 | vehicle | 0.9553 | 0.9583 | +0.0030 | 80.85% | 82.03% | +1.18% |
| 13 | eucalyptus | 0.9109 | 0.9138 | +0.0029 | 66.30% | 67.93% | +1.63% |
| 14 | analcatdata_authorship | 0.9998 | 1.0000 | +0.0002 | 99.29% | 99.76% | +0.47% |
| 15 | analcatdata_dmft | 0.5512 | 0.5572 | +0.0060 | 19.05% | 19.55% | +0.50% |
| 16 | pc4 | 0.9274 | 0.9340 | +0.0066 | 89.80% | 90.20% | +0.40% |
| 17 | pc3 | 0.8286 | 0.8427 | +0.0141 | 90.20% | 91.00% | +0.80% |
| 18 | kc2 | 0.8217 | 0.8225 | +0.0008 | 84.29% | 83.91% | -0.38% |
| 19 | pc1 | 0.8711 | 0.8799 | +0.0088 | 93.20% | 92.80% | -0.40% |
| 20 | banknote-authentication | 1.0000 | 1.0000 | +0.0000 | 100.00% | 100.00% | +0.00% |
| 21 | blood-transfusion-service | 0.7897 | 0.7907 | +0.0010 | 79.14% | 78.61% | -0.53% |
| 22 | ilpd | 0.7441 | 0.7370 | -0.0071 | 73.29% | 73.97% | +0.68% |
| 23 | qsar-biodeg | 0.9368 | 0.9371 | +0.0003 | 88.20% | 88.00% | -0.20% |
| 24 | wdbc | 0.9982 | 0.9984 | +0.0002 | 98.60% | 98.25% | -0.35% |
| 25 | cylinder-bands | 0.7656 | 0.7673 | +0.0017 | 71.11% | 70.74% | -0.37% |
| 26 | dresses-sales | 0.5331 | 0.5327 | -0.0004 | 56.80% | 56.80% | +0.00% |
| 27 | MiceProtein | 0.9942 | 0.9982 | +0.0040 | 92.00% | 96.20% | +4.20% |
| 28 | car | 0.7886 | 0.8184 | +0.0298 | 78.60% | 77.40% | -1.20% |
| 29 | steel-plates-fault | 0.9269 | 0.9385 | +0.0116 | 69.40% | 71.80% | +2.40% |
| 30 | climate-model-simulation | 0.9437 | 0.9444 | +0.0007 | 94.07% | 94.44% | +0.37% |
| **Σ** | **OVERALL MEAN** | **0.8668** | **0.8706** | **+0.0038** | **79.27%** | **80.12%** | **+0.85%** |

---

Clearly, as also stated in the paper. Within the datasets with lots of categorical features may lead TabPFN to predict worse. 

## Quickstart & Inference

### 1. Installation
```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy tqdm matplotlib

python benchmark.py
```

```bash
import torch
from inference import predict
from train import Transformer_TabPFN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer_TabPFN().to(device)
checkpoint = torch.load("tabpfn_step_18000.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Single forward pass zero-shot prediction
# X_train: [N_train, k], y_train: [N_train], X_test: [N_test, k]
predictions = predict(model, X_train, y_train, X_test, n_ensemble=32) # includes all normalization stuff here
```

## References
```bibtext
@inproceedings{hollmann2023tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Noah Hollmann and Samuel M{\"u}ller and Katharina Eggensperger and Frank Hutter},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```
