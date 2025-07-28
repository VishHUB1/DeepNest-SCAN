# DeepNest-SCAN
DeepNest-SCAN: Unsupervised crack detection using nested autoencoders, trained on non-cracked surfaces. Achieves 0.93 accuracy on METU dataset and 0.98 on real-time UAV images. Includes SDNET2018/METU analysis with BIRCH/UMAP, supervised baselines (DenseNet121, ResNet50, etc.), and scalable real-time monitoring.
DeepNest-SCAN: Unsupervised Surface Crack Anomaly Detection

Overview
DeepNest-SCAN is an unsupervised deep learning framework designed for detecting surface cracks in concrete structures using nested autoencoders. Unlike traditional supervised methods that rely on large labeled datasets, DeepNest-SCAN is trained exclusively on non-cracked surfaces to identify anomalous patterns indicative of cracks. This approach enhances its adaptability to unseen and irregular crack patterns, making it suitable for real-time infrastructure monitoring.
The project leverages datasets such as SDNET2018 and METU, achieving a validation accuracy of 0.93 on the METU dataset and a real-time accuracy of 0.98 with UAV-captured images. The framework is lightweight, scalable, and robust, addressing challenges like environmental variability and computational constraints.
Features

Unsupervised Learning: Trained on non-cracked surfaces to detect cracks as anomalies, eliminating the need for extensive labeled datasets.
High Accuracy: Achieves 0.93 accuracy on METU dataset and 0.98 on real-time UAV data.
Real-Time Deployment: Fine-tuned for UAV-captured images, enabling scalable and efficient crack detection.
Clustering Analysis: Utilizes BIRCH and UMAP for dataset exploration, revealing data organization and crack severity insights.
Future Potential: Plans for a Crack Severity Index to prioritize repair efforts based on anomaly scores.

Repository Structure
DeepNest-SCAN/
├── data/
│   ├── SDNET2018/         # SDNET2018 dataset (not included, see below)
│   ├── METU/              # METU dataset (not included, see below)
│   └── UAV/               # UAV-captured images for real-time testing
├── models/
│   ├── pretrained/        # Pre-trained CNN models (DenseNet121, ResNet50, etc.)
│   └── deepnest_scan/     # DeepNest-SCAN autoencoder model
├── scripts/
│   ├── data_processing.py # Scripts for dataset loading and preprocessing
│   ├── clustering.py      # BIRCH and UMAP clustering scripts
│   ├── train_supervised.py # Training script for supervised CNN models
│   ├── train_deepnest.py  # Training script for DeepNest-SCAN
│   ├── evaluate.py        # Evaluation script for model performance
│   └── real_time.py       # Real-time crack detection with UAV images
├── notebooks/
│   ├── exploratory_analysis.ipynb # Jupyter notebook for dataset analysis
│   └── model_training.ipynb      # Jupyter notebook for model training
├── media/                 # Images and figures for documentation
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── LICENSE                # License file

Installation
Prerequisites

Python 3.8+
CUDA-enabled GPU (recommended for training)
Git

Steps

Clone the Repository:
git clone https://github.com/your-username/DeepNest-SCAN.git
cd DeepNest-SCAN


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

The requirements.txt includes:
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
umap-learn>=0.5.0
matplotlib>=3.4.0
opencv-python>=4.5.0


Download Datasets:

SDNET2018: Download from University of Utah and place in data/SDNET2018/.
METU: Obtain from METU repository and place in data/METU/.
UAV Images: Collect or use provided sample UAV images in data/UAV/.



Usage
1. Dataset Exploration
Run the clustering analysis to understand the dataset structure:
python scripts/clustering.py --dataset METU --k 2

This generates BIRCH and UMAP visualizations (saved in media/).
2. Supervised Training
Train pre-trained CNN models (DenseNet121, ResNet50, VGG16, InceptionV3) on the METU dataset:
python scripts/train_supervised.py --model DenseNet121 --dataset METU

Results are saved in models/pretrained/ and visualizations in media/.
3. DeepNest-SCAN Training
Train the unsupervised nested autoencoder on non-cracked METU images:
python scripts/train_deepnest.py --dataset METU --non-cracked-only

Model checkpoints are saved in models/deepnest_scan/.
4. Evaluation
Evaluate DeepNest-SCAN performance using 5-fold cross-validation:
python scripts/evaluate.py --model deepnest_scan --dataset METU

Metrics and confusion matrices are saved in media/.
5. Real-Time Crack Detection
Test DeepNest-SCAN on UAV-captured images:
python scripts/real_time.py --model deepnest_scan --uav-data data/UAV/

Results include real-time accuracy and reconstruction error distributions.
Jupyter Notebooks
Explore the dataset and model training interactively:
jupyter notebook notebooks/exploratory_analysis.ipynb
jupyter notebook notebooks/model_training.ipynb

Model Architecture
DeepNest-SCAN uses a nested autoencoder with the following structure:

Input: RGB images (128x128x3)
Encoder:
3 Conv2D layers (filters: 32, 64, 128; ReLU activation)
2 MaxPooling2D layers (pool size: 2x2)
Flatten layer
3 Dense layers (units: 512, 256, 128; bottleneck at 128)


Decoder:
3 Dense layers (units: 128, 256, 512)
Reshape layer (target shape: 32x32x128)
2 UpSampling2D layers (size: 2x2)
3 Conv2D layers (filters: 128, 64, 32; sigmoid activation for output)



Training Parameters:

Batch Size: 32
Epochs: 50
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Early Stopping: Enabled (patience: 10)
Device: GPU (CUDA)

Performance Metrics
Supervised Models (METU Dataset)



Model
Accuracy
Precision
Recall
F1-Score



DenseNet121
0.9982
0.9985
0.9980
0.9982


InceptionV3
0.9978
0.9965
0.9992
0.9978


VGG16
0.9964
0.9940
0.9988
0.9964


ResNet50
0.9694
0.9925
0.9460
0.9687


EfficientNetB0
0.5000
0.0000
0.0000
0.0000


DeepNest-SCAN (METU Dataset, 5-Fold Cross-Validation)



Iteration
Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC



1
0.9169
0.9040
0.932
0.918
0.9717
0.967


2
0.9307
0.9241
0.938
0.931
0.9795
0.974


3
0.9273
0.9109
0.947
0.928
0.9776
0.973


4
0.9126
0.9160
0.908
0.912
0.9683
0.962


5
0.9274
0.9136
0.944
0.928
0.9780
0.973


Avg
0.9230
0.9137
0.934
0.923
0.9750
0.970


Real-Time (UAV Data)



Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
Threshold



0.98
1.0
0.96
0.9796
1.0
1.0
0.00146


Datasets

SDNET2018:

56,000+ annotated images (256x256, RGB)
Categories: Decks, Pavements, Walls (cracked/non-cracked)
Crack widths: 0.06–0.25 mm
Source: University of Utah


METU:

40,000 images (277x277, RGB; 20,000 cracked, 20,000 non-cracked)
Derived from 458 high-resolution images (4032x3024)
Source: METU campus buildings


UAV Data:

500 non-cracked images for fine-tuning
Used for real-time crack detection



Contributing
We welcome contributions to improve DeepNest-SCAN! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Citation
If you use DeepNest-SCAN in your research, please cite:
Vishal Ra, Deepak Kb, Venkatesan Kb, Syarifah Bahiyah Rahayu, Najah Alsubaie.
"DeepNest-SCAN: An Unsupervised Surface Crack Anomaly Detection using Nested Autoencoders."

Contact
For inquiries, contact the corresponding author at syarifahbahiyah@upnm.edu.my.
