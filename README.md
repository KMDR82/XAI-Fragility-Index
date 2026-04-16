# Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
[![DOI](https://zenodo.org/badge/1212214151.svg)](https://doi.org/10.5281/zenodo.19604732)

## 1. Description
This repository contains the official codebase and implementation for the paper: **"Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging: A Novel Index for Assessing Heatmap Stability"**. 

The objective of this study is to quantitatively evaluate the structural robustness of Explainable Artificial Intelligence (XAI) methods in clinical decision support systems. We propose a novel metric, the **XAI Fragility Index (XFI)**, which mathematically measures the degradation of XAI heatmaps under real-world radiological corruptions. The framework evaluates Grad-CAM and Eigen-CAM using a ResNet-50 classifier under strictly controlled stress tests including Gaussian Noise, Gaussian Blur, and Salt-and-Pepper perturbations.

## 2. Dataset Information
The model was trained and evaluated using the publicly available **RSNA Pneumonia Detection Challenge** dataset.
* **Total Patients:** 26,684 clinical chest X-ray (CXR) images.
* **Data Splitting:** The dataset is split into 80% training and 20% testing sets at the patient ID level to strictly prevent data leakage.
* **Labels:** Binary classification (Pneumonia vs. Normal) with bounding box coordinates for positive cases to evaluate localization shift (Intersection over Union - IoU).

## 3. Repository Structure
The repository is structured to ensure end-to-end reproducibility:
* `main.py`: The primary orchestration script. Handles dataset loading, model inference, perturbation pipeline, and XFI computation.
* `visualize.py`: Script to generate publication-ready (300 DPI) figures from the experimental results.
* `plots/`: Contains all high-resolution figures and charts presented in the manuscript.
* `logs/`: Contains the comprehensive experimental results. Large CSV files are provided in a compressed format (`.zip`) to comply with repository size constraints.
* `requirements.txt`: Python dependencies.

## 4. Requirements
The project is built using Python 3.12. Required libraries:
* `torch` (>= 2.0.0), `torchvision` (>= 0.15.0), `numpy`, `pandas`, `opencv-python`, `pydicom`, `scikit-learn`, `scipy`, `scikit-image`, `grad-cam`.

## 5. Usage Instructions
Follow these steps to replicate the study:

**Step 1: Data and Model Preparation**
* **Dataset:** Download the RSNA dataset from [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge).
* **Model Weights:** Download `best_pneumonia_model.pth` from [this Google Drive link](https://drive.google.com/file/d/1Ae31_XiP_SsvWOnnItbOiYKPmTH31b_o/view?usp=sharing) and place it in the root directory.

**Step 2: Environment Setup**
```bash
pip install -r requirements.txt
```

**Step 3: Execution**
Run the main experiment script (Global seed `42` is enforced):
```bash
python main.py
```

**Step 4: Visualization and Results**
* To regenerate figures: `python visualize.py`
* **Note:** Raw experimental data and pre-generated figures are available in the `/logs` and `/plots` directories respectively.

## 6. Citation
If you use this dataset, codebase, or the XFI metric in your research, please cite our paper (currently under review) and the Zenodo archive:

> A. Akkaya, "Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging: A Novel Index for Assessing Heatmap Stability", *Submitted for publication*, 2026. 
> Code archive: [![DOI](https://zenodo.org/badge/1212214151.svg)](https://doi.org/10.5281/zenodo.19604732)

*(Note: The full citation details will be updated upon the paper's acceptance and publication).*

## 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
