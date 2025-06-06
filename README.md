# LCTC: Lightweight Convolutional Thresholding Sparse Coding Network Prior for Computational Hyperspectral Imaging

This repository contains the official implementation of **LCTC (Lightweight Convolutional Thresholding Coding)**, a self-supervised hyperspectral image reconstruction framework that integrates **convolutional sparse coding** into an efficient deep prior for compressive hyperspectral imaging systems (e.g., CASSI, hyperspectral denoising, etc.).

<p align="center">
  <img src="./lctc_pipeline.jpg" alt="LCTC pipeline overview" width="600"/>
</p>

---

## 🔍 Overview

**LCTC** (Lightweight Convolutional Thresholding sparse Coding) is a novel *untrained neural network prior* designed for **compressive spectral imaging**. Inspired by convolutional sparse coding theory, LCTC treats the network as a *learnable sparse transform domain* and its input as the corresponding *sparse coefficients*. Unlike traditional model-based approaches that rely on handcrafted priors or deep learning methods that require large-scale training data, LCTC learns both the sparse domain and signal representation in a **self-supervised** manner directly from the compressed measurements.

LCTC can be deployed in two modes:

* **Standalone Prior:** Used independently for hyperspectral image reconstruction without external optimization.
* **PnP-LCTC:** Incorporated into iterative optimization frameworks (e.g., ADMM) as a **Plug-and-Play** regularizer to further boost reconstruction performance.

---

## ✨ Key Features

* **Model-Inspired Untrained Prior:** The LCTC network mimics convolutional sparse coding, providing an interpretable prior that bridges model-based and learning-based methods.

* **Self-Supervised Sparse Learning:** Both the transform domain and sparse coefficients are learned from a single measurement, eliminating the need for training datasets.

* **Lightweight & Efficient:** The network contains only \~0.24MB parameters and requires 4.7 GFLOPs, enabling fast inference and practical deployment in real-time or resource-limited settings.

* **Plug-and-Play Compatibility:** LCTC can be integrated into existing iterative solvers (e.g., ADMM) as a regularization prior, enhancing flexibility and performance.

* **Robust & High-Fidelity:** Outperforms existing untrained and model-based methods in terms of reconstruction accuracy, noise robustness, and generalization across different compressive spectral imaging systems.

* **Versatile Applicability:** Validated on both **snapshot compressive spectral imaging** (e.g., CASSI) and **line-scan hyperspectral imaging**, demonstrating strong adaptability to diverse imaging scenarios.

---

## 🧠 Network Architecture

The **LCTC** (Lightweight Convolutional Thresholding sparse Coding) network is a novel **untrained neural network prior** specifically designed for compressive spectral imaging (CSI). It blends model-based interpretability with the flexibility of deep learning by learning both the sparse transform domain and the compressible signal directly from encoded measurements—without any external training data.

### 🔧 Core Design: Convolutional Thresholding Sparse Coding (CTC)

At the heart of LCTC lies the **CTC module**, which simulates the sparse coding process by:

* **Convolutional dictionaries**: Three efficient convolutional layers (with kernel sizes 3×3, 3×1, and 1×3) extract meaningful features while minimizing computational load.
* **Nonlinear thresholding**: A learnable soft-thresholding mechanism selects important spectral features based on global pooling and fully connected layers, producing sparse activations.
* **Residual learning**: Residual connections enhance convergence and optimization efficiency.

These CTC modules transform feature maps into sparse representations and back, forming the basis of a learnable sparse transformation.

Despite its expressive capability, the LCTC network remains extremely compact:

* **Model size**: \~0.24MB
* **Computational cost**: 4.7 GFLOPs

This design enables fast and memory-efficient inference suitable for real-time or embedded spectral imaging applications.

---


## 📁 Directory Structure

```bash
LCTC/
├── models/                # Network architecture definitions
│   └── lctc_net.py
├── data/                  # Scripts for loading hyperspectral datasets
│   └── load_kaist.py
├── utils/                 # Utility functions for metrics, visualization, etc.
│   └── metrics.py
├── configs/               # Configurations and parameters
│   └── lctc_config.yaml
├── train.py               # Training script (unsupervised)
├── test.py                # Evaluation on benchmark datasets
├── forward_model.py       # Forward operator (CASSI, SFC, etc.)
└── README.md              # Project description
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
conda create -n lctc_env python=3.8
conda activate lctc_env
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download and preprocess datasets:

* [KAIST CASSI Dataset](https://github.com/HubertWong0501/KAIST-CASSI)
* [WDC HYDICE Dataset](https://www.sdms.afrl.af.mil/index.php?collection=hyperspectral)

Modify `configs/lctc_config.yaml` to point to the dataset directory.

### 3. Train the Model

```bash
python train.py --config configs/lctc_config.yaml
```

This script trains LCTC in a self-supervised manner using only compressed measurements.

### 4. Test the Model

```bash
python test.py --ckpt checkpoints/lctc_kaist.pth --dataset KAIST
```

---

## 📊 Evaluation Metrics

We report standard hyperspectral reconstruction metrics:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* SAM (Spectral Angle Mapper)

Evaluation results on public datasets can be reproduced using provided scripts.

---

## 📎 Citation

If you use LCTC in your research, please cite:

```bibtex
@article{your_lctc_paper,
  title={LCTC: Lightweight Convolutional Thresholding Sparse Coding Network Prior for Computational Hyperspectral Imaging},
  author={Your Name and Collaborators},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
}
```

---

## 🤝 Acknowledgments

This work was supported by \[Your Funding Agency]. Some code components are adapted from DIP and DeSCI repositories.

---

## 📬 Contact

For questions, please contact:

**\[陈煜嵘 (Yurong Chen)]**
*PhD Candidate*
Email: \[[your.email@example.com](mailto:your.email@example.com)]

---

Let me know if you'd like the README customized for a particular codebase structure, dataset, or if you want to include image illustrations (e.g., training loss curves or result comparisons).
