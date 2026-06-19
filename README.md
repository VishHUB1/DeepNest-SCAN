📋 Overview

DeepNest-SCAN is a novel unsupervised crack detection system that overcomes the limitations of supervised approaches requiring extensive labeled datasets. The model uses nested autoencoders trained exclusively on non-cracked surfaces to detect anomalous crack patterns in concrete infrastructure.

Key Advantages

✅ No labeled crack data required — trains on non-cracked surfaces only

✅ Real-time inference — 98% accuracy on UAV-captured images

✅ Generalizable — detects unseen crack patterns without overfitting

✅ Scalable — lightweight architecture for edge deployment

✅ Production-ready — validated across multiple datasets and environmental conditions


🎯 Performance

DatasetAccuracyPrecisionRecallF1-ScoreROC-AUCMETU (5-fold CV)92.30%91.37%93.40%92.30%0.9750UAV Real-time98.00%100%96%0.97961.00

Baseline Comparison (Transfer Learning on METU)


DenseNet121: 99.82% accuracy (supervised, requires labeled data)
InceptionV3: 99.78% accuracy (supervised, requires labeled data)
VGG16: 99.64% accuracy (supervised, requires labeled data)


DeepNest-SCAN achieves 93% accuracy on METU with zero labeled crack annotations, making it ideal for generalizing to unseen surfaces.


🏗️ Architecture

Nested Autoencoder Design

Input (128×128×3)
    ↓
[Encoder] Convolutional Layers
  - Conv2D (filters: 32, 64, 128) + ReLU
  - MaxPooling2D (pool_size: 2×2)
  - Flatten
  - Dense layers (512 → 256 → 128 units)
    ↓
[Bottleneck] Latent Space (128 units)
    ↓
[Decoder] Dense Layers
  - Dense layers (128 → 256 → 512 units)
  - Reshape (32×32×128)
  - Upsampling2D (size: 2×2)
  - Conv2D (filters: 128, 64, 32) + Sigmoid
    ↓
Output (128×128×3)

Detection Mechanism


Training: Learns reconstruction patterns from non-cracked images
Testing: High reconstruction error (MSE) → Anomaly (Crack)
Thresholding: Dynamic threshold set during training
Severity: MSE value correlates with crack severity



📦 Installation

Requirements


Python 3.8+
TensorFlow 2.4+
NumPy, scikit-learn, OpenCV, Matplotlib
