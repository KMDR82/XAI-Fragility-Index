# XAI Fragility Index (XFI): model weights, per-patient results and analysis tables (release 2.0)

Data deposit accompanying the article "Quantifying the structural fragility of explainable artificial intelligence
in medical imaging: A novel index for assessing heatmap stability" (Ahmet Akkaya, submitted to PeerJ Computer Science).
The code that produced every file is in https://github.com/KMDR82/XAI-Fragility-Index, release v2.0.0,
which is archived on Zenodo as a new version of https://doi.org/10.5281/zenodo.19604732.

The RSNA Pneumonia Detection Challenge images are not included; they can be obtained from
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge under the competition terms.
The stored example maps of the runs are not included either, because they contain image crops.

## Files

| Path | Content |
|---|---|
| `model_A.pth` | Model A: ResNet-50 trained from scratch for 5 epochs, no ImageNet normalisation, threshold 0.5 |
| `model_Arep.pth` | Model A-rep: same recipe as A, trained on the training set without the validation split |
| `model_Anorm.pth` | Model A-norm: as A-rep, with ImageNet normalisation |
| `model_B.pth` | Model B: ImageNet-pretrained ResNet-50, positive-class weight, early stopping on validation AUC |
| `zenodo_results_tables.zip` | everything below; `CONTENTS_SHA256.txt` inside the zip lists the checksums of its files |
| `results/xfi_<model>/results_main.csv.gz` | one row per test patient x method x perturbation condition |
| `results/xfi_<model>/results_base.csv.gz` | one row per test patient x baseline: `Random_Pixel`, `Random_Smooth`, `Uniform` (constant map), `CrossPatient_<method>` (`partner` = the other patient), `ModelRandom_<method>` (partially randomised model) |
| `results/xfi_<model>/manifest.json` | run settings: weights SHA-256, decision threshold, config hash, outcome group sizes, software versions (the `git` field is the development commit of the run; the same pipeline is published as release v2.0.0) |
| `splits/` | frozen patient-level split: `split_train.csv`, `split_test.csv` (5,337 patients), `split_val.csv` (validation patients held out from the training set) and `split_train_fit.csv` (training set without them) |
| `training/train_<model>/` | per-epoch validation metrics, validation and test probabilities, training summary (A-rep, A-norm, B) |
| `tables/R*.csv` | analysis tables behind the manuscript (R1 classifier performance ... R18 XFI by prediction outcome) |
| `config.yaml`, `requirements-r1.txt` | configuration and pinned package versions used for the run; in `config.yaml` the path of a local backup folder (`drive_backup_dir`), which does not affect any result, is replaced by `SET_ME`, so its SHA-256 differs from `config_sha256` in the manifests |
| `SHA256SUMS.txt` | SHA-256 checksums of the files of this record (`sha256sum -c SHA256SUMS.txt` in the download folder) |

All weights are PyTorch `state_dict`s of `torchvision.models.resnet50` with a single-logit output layer
(`fc = Linear(2048, 1)`); inputs are 224 x 224 three-channel images (the grey value replicated), scaled to [0, 1],
with ImageNet mean/std normalisation for A-norm and B only.

## Columns of `results_main`

| Column | Meaning |
|---|---|
| `Patient`, `Label` | RSNA patient ID; 1 = pneumonia (at least one bounding box), 0 = no box |
| `prob_clean`, `pred_clean`, `threshold` | predicted probability and class on the clean image, decision threshold of the model |
| `Method` | `GradCAM`, `EigenCAM`, `EigenCAM_SC` (Eigen-CAM with sign correction, sensitivity analysis) |
| `Type`, `Level` | perturbation (`Gaussian` noise, Gaussian `Blur`, `SaltPepper`) and severity 1-3 |
| `prob_noisy`, `pred_noisy`, `flip` | prediction on the perturbed image; `flip` = 1 if the class changed |
| `input_psnr`, `input_ssim` | PSNR and SSIM between the clean and the perturbed input image |
| `ssim`, `p_corr`, `s_corr`, `rs`, `xfi` | SSIM and Pearson / Spearman correlation between the clean and the perturbed map, robustness score RS and XFI = 1 - RS |
| `degenerate` | 1 if a map was constant (correlations undefined; convention described in the article) |
| `iou_rel80_*`, `iou_rel90_*` | IoU with the ground-truth box mask, map thresholded at its 80th / 90th percentile (`_c` clean, `_n` perturbed) |
| `iou_abs5_*`, `iou_abs7_*` | IoU with the box mask at the absolute thresholds 0.5 / 0.7 of the min-max normalised map |
| `far_bg_abs5_*`, `far_bg_abs7_*` | share of the pixels outside the boxes whose activation reaches 0.5 / 0.7 |
| `prec_in_box_abs5_*`, `prec_in_box_abs7_*` | share of the activated pixels that lie inside the boxes |
| `energy_out_box_*` | share of the activation energy outside the boxes |
| `act_area_abs5_*`, `act_area_abs7_*` | share of all pixels whose activation reaches 0.5 / 0.7 (used for negative patients) |
| `com_drift_px` | shift of the activation centre of mass between the clean and the perturbed map (pixels) |
| `eig_*` | diagnostics of the sign of the Eigen-CAM principal component |

Localization columns are empty for negative patients.

## License

The weights, results and tables are released under CC BY 4.0. The code is MIT-licensed (see the code archive).
