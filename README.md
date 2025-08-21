# Brain Tumor Classification using CNNs & Pretrained Models

This repository provides a pipeline to **classify brain tumors from MRI scans** using deep learning, including **custom CNN architectures (from scratch)** and **fine-tuned pretrained models like MobileNet**.

---

## Table of Contents
- [Motivation](#motivation)
- [Features](#features)
- [Methodology](#methodology)
- [Models](#models)
- [Installation](#installation)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Motivation

Accurate classification of brain tumors—such as glioma, meningioma, and pituitary tumors—from MRI scans is critical for early diagnosis and treatment planning. Manual interpretation is time-consuming and prone to variability. This project leverages deep learning to automate and improve diagnostic accuracy.

---

## Features

- **CNN from scratch**: Custom lightweight model for classification.
- **Pretrained MobileNet**: Transfer learning for faster convergence and higher accuracy.
- **Evaluation Metrics**: Confusion matrix, precision, recall, F1-score.
- **Data Augmentation**: Improves robustness and generalization.
- **Visualization**: Loss/accuracy curves and prediction examples.

---

## Methodology

1. **Data Loading & Preprocessing**
   - MRI scans resized to a fixed resolution (224×224).
   - Pixel normalization for faster convergence.
   - Augmentation (flips, rotations, brightness) to prevent overfitting.

2. **Model Training**
   - Train CNN from scratch as a baseline.
   - Fine-tune MobileNet for improved accuracy.
   - Optimize using Adam with learning rate scheduling.

3. **Evaluation**
   - Use confusion matrix + classification report.
   - Compare CNN vs MobileNet results.
   - Visualize results with prediction examples.

---

## 🏗Models

### 1. CNN (From Scratch)
A custom convolutional neural network built from the ground up.  
**Architecture:**
- **Conv Layers**: Learn spatial patterns in MRI scans.
- **ReLU Activation**: Introduces non-linearity.
- **MaxPooling**: Reduces spatial dimensions.
- **Dropout**: Prevents overfitting.
- **Fully Connected Layers**: Final classification into tumor categories.

This model serves as a **baseline** to understand performance without transfer learning.

---

### 2. MobileNet (Pretrained)
MobileNet is a lightweight convolutional neural network optimized for mobile and embedded vision applications.  
Here, it is **fine-tuned** for brain tumor classification:

- **Base Layers (Frozen initially)**: Pretrained on ImageNet to extract rich image features.
- **Custom Classifier Head**:
  - Global Average Pooling
  - Dense Layer with Softmax activation for classification.
- **Advantages**:
  - Faster training compared to scratch models.
  - Higher accuracy due to transfer of pre-learned features.
  - Efficient and less memory intensive.

---

## Installation

```bash
git clone https://github.com/Soumit0702/Brain_tumor_classification.git
cd Brain_tumor_classification

pip install torch torchvision matplotlib scikit-learn
