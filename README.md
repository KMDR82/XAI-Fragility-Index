\# Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging



\[!\[PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)

\[!\[Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)

\[!\[License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)



\## 1. Description

This repository contains the official codebase and implementation for the paper: \*\*"Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging: A Novel Index for Assessing Heatmap Stability"\*\*. 



The objective of this study is to quantitatively evaluate the structural robustness of Explainable Artificial Intelligence (XAI) methods in clinical decision support systems. We propose a novel metric, the \*\*XAI Fragility Index (XFI)\*\*, which mathematically measures the degradation of XAI heatmaps under real-world radiological corruptions. The framework evaluates Grad-CAM and Eigen-CAM using a ResNet-50 classifier under strictly controlled stress tests including Gaussian Noise, Gaussian Blur, and Salt-and-Pepper perturbations.



\## 2. Dataset Information

The model was trained and evaluated using the publicly available \*\*RSNA Pneumonia Detection Challenge\*\* dataset.

\* \*\*Total Patients:\*\* 26,684 clinical chest X-ray (CXR) images.

\* \*\*Data Splitting:\*\* The dataset is split into 80% training and 20% testing sets at the patient ID level to strictly prevent data leakage.

\* \*\*Labels:\*\* Binary classification (Pneumonia vs. Normal) with bounding box coordinates for positive cases to evaluate localization shift (Intersection over Union - IoU).



\## 3. Repository Structure

The repository is structured to ensure end-to-end reproducibility:

\* `main.py`: The primary orchestration script. It handles dataset loading, ResNet-50 training, the perturbation pipeline (stress testing), XAI generation (Grad-CAM, Eigen-CAM), and computes the XFI metric.

\* `visualize.py`: \*(To be added)\* Scripts for generating publication-ready figures, including XFI distribution violin plots, heatmap degradation grids, and correlation matrices.

\* `requirements.txt`: Contains all necessary Python dependencies.



\## 4. Requirements

The project is built using Python 3.12. The required libraries to run the code are listed below:

\* `torch` (>= 2.0.0)

\* `torchvision` (>= 0.15.0)

\* `numpy`

\* `pandas`

\* `opencv-python` (`cv2`)

\* `pydicom`

\* `scikit-learn`

\* `scipy`

\* `scikit-image`

\* `grad-cam` (`pytorch-grad-cam`)



\## 5. Usage Instructions

Follow these steps to replicate the study and generate the XFI metrics:



\*\*Step 1: Data Preparation\*\*

\* Download the RSNA Pneumonia Detection Challenge dataset from Kaggle.

\* Ensure the `stage\_2\_train\_labels.csv` and the image directory (containing `.dcm`, `.png`, or `.jpg` files) are located in your working directory or input path.



\*\*Step 2: Execution\*\*

Run the main experiment script. To ensure PeerJ reproducibility standards, a global deterministic seed (`42`) is strictly enforced within the script.

```bash

python main.py

