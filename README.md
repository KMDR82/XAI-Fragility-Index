# Quantifying the Structural Fragility of Explainable Artificial Intelligence in Medical Imaging

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![DOI](https://zenodo.org/badge/1212214151.svg)](https://doi.org/10.5281/zenodo.19604732)

Code for the article **"Quantifying the structural fragility of explainable artificial intelligence in medical
imaging: A novel index for assessing heatmap stability"** (Ahmet Akkaya, under review at PeerJ Computer Science).

The **XAI Fragility Index (XFI)** measures how much an explanation heatmap changes when the input image is perturbed:

    RS  = (SSIM + r_p^norm + r_s^norm) / 3,   r^norm = (r + 1) / 2,   XFI = 1 - RS,   XFI in [0, 4/3]

computed between the min-max normalised maps of the clean and the perturbed image. Unrelated maps give 2/3.
The study applies XFI to Grad-CAM and Eigen-CAM explanations of four ResNet-50 pneumonia classifiers on the
RSNA Pneumonia Detection Challenge test set (5,337 patients) under Gaussian noise, Gaussian blur and
salt-and-pepper noise at three levels, and relates it to input degradation, localization against the expert
boxes, prediction flips and five baselines.

## Release 2.0.0 (revision)

| Archive | Content | DOI |
|---|---|---|
| Code (this repository, v2.0.0) | pipeline, tests, notebooks, figures with their data | CODE_DOI |
| Data | weights of the four classifiers, per-patient results, frozen splits, training logs, analysis tables, SHA-256 checksums | DATA_DOI |

Version 1.0.0 (the code of the original submission, `main.py` / `visualize.py`) remains available under the tag
[`v1.0.0`](https://github.com/KMDR82/XAI-Fragility-Index/tree/v1.0.0) and on Zenodo
([10.5281/zenodo.19604732](https://doi.org/10.5281/zenodo.19604732)).

## Classifiers

| Model | Recipe | Decision threshold |
|---|---|---|
| A | ResNet-50 from scratch, 5 epochs, no ImageNet normalisation, unweighted loss | 0.5 |
| A-rep | same recipe, trained on the training set without the validation split | validation (Youden) |
| A-norm | as A-rep, with ImageNet normalisation | validation (Youden) |
| B | ImageNet-pretrained, positive-class weight, early stopping on validation AUC | validation (Youden) |

## Repository structure

```
configs/config.yaml       central configuration (paths, seed, split, perturbation levels)
src/xfi/                  pipeline: data.py (split, image cache), perturb.py, xai.py (Grad-CAM, Eigen-CAM),
                          metrics.py (XFI, IoU, background activation), train.py, run_xfi.py
src/r1lib/                statistics (patient-level tests, bootstrap, Holm) and figure style
tests/                    unit tests (pytest)
notebooks/                00 environment check, 01 audit of the version 1.0.0 results, 02 pilot,
                          03 analysis (tables R1-R13), 04 figures, 05 sensitivity analyses (R14-R18)
scripts/                  environment setup, RSNA download, training and XFI runs, packaging of the data deposit
fig/manuscript/           figures of the article (PNG, PDF, 300 dpi) and the data of every figure (CSV)
```

## Reproducing the results

1. **Environment** (Linux or WSL2, CUDA GPU):
   ```bash
   bash scripts/00_setup_env.sh            # conda env "xfi": torch 2.6.0 / torchvision 0.21.0 (CUDA 12.4) + requirements-r1.txt
   PYTHONPATH=src pytest -q tests
   ```
2. **Data**: download the RSNA Pneumonia Detection Challenge data from
   [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (`bash scripts/01_download_rsna.sh`,
   needs a Kaggle API token) into `dataset/rsna/`.
3. **Splits and weights**: from the data deposit (DATA_DOI), copy `splits/split_train.csv` and `split_test.csv` to
   `output/phase0/`, `split_val.csv` and `split_train_fit.csv` to `output/phase2/`, and the weights to `model/`
   (`model_A.pth` -> `modelA_kaggle_v1.pth`, `model_Arep.pth` -> `Arep_best.pth`, `model_Anorm.pth` -> `Anorm_best.pth`,
   `model_B.pth` -> `B_best.pth`), and the `training/train_<model>/summary.json` files to `output/phase2/train_<model>/`
   (they hold the decision thresholds). Check the downloaded files with `sha256sum -c SHA256SUMS.txt`, and the unpacked zip with `sha256sum -c CONTENTS_SHA256.txt`.
   To retrain instead: `PYTHONPATH=src python -m xfi.train --model Arep` (likewise `Anorm`, `B`).
4. **XFI runs** (one per classifier; interrupted runs resume, every random draw is keyed by seed, patient and condition):
   ```bash
   PYTHONPATH=src python -m xfi.run_xfi --model A      # likewise Arep, Anorm, B
   ```
5. **Analysis and figures**: run `notebooks/03_analysis.ipynb`, `04_figures.ipynb` and `05_sensitivity.ipynb`.
   The per-patient results and tables of the deposit can be used directly instead of steps 3-4.

Global seed: 42. Deterministic cuDNN settings are enforced in training and inference.

## Citation

> A. Akkaya, "Quantifying the structural fragility of explainable artificial intelligence in medical imaging:
> A novel index for assessing heatmap stability", submitted to *PeerJ Computer Science*, 2026.
> Code: CODE_DOI. Data: DATA_DOI.

See also `CITATION.cff`. The citation will be updated on publication.

## License

Code: MIT (see `LICENSE`). Data deposit: CC BY 4.0. The RSNA images are subject to the terms of the Kaggle competition.
