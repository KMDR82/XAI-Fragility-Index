# Figure captions (draft, v2 data)

## Fig01_framework

Evaluation framework. Every test image and its nine perturbed versions are classified; explanations are computed for the class predicted on the clean image and compared between the clean and each perturbed version with the XAI Fragility Index (XFI), with localization against the radiologist's box, with the degradation of the input, and against baselines. Statistics use the patient as the unit of analysis.

## Fig02_classifiers

Classifier performance on the 5,337 test patients. (a) ROC curves and (b) reliability diagrams (10 equal-width bins) of Model A (the classifier of the submitted study), Model A-rep (same recipe, retrained), Model A-norm (same recipe with ImageNet normalisation) and Model B (ImageNet-pretrained, class-weighted loss). (c) Validation AUC and (d) validation expected calibration error per epoch for Model A-rep and Model A-norm, which differ only in input normalisation.

## Fig03_kinetics

Mean XFI against perturbation severity for Grad-CAM and Eigen-CAM. Rows: classifiers; columns: Gaussian noise (sigma = 0.1, 0.2, 0.3), Gaussian blur (kernel 3, 5, 7) and salt-and-pepper noise (1, 2, 3 % of pixels). Shaded bands: 95 % patient-bootstrap confidence intervals. Dashed lines: cross-patient XFI of the same method and classifier, i.e. the typical dissimilarity between the clean explanations of two different patients.

## Fig04_grad_vs_eigen

Median paired difference in XFI between Grad-CAM and Eigen-CAM (Grad-CAM minus Eigen-CAM) per perturbation condition and for the patient-level mean over all conditions, for each classifier. Bars: 95 % patient-bootstrap CIs. Values below zero: Grad-CAM less fragile. Colour: method with the lower XFI where the paired Wilcoxon test is significant after Holm correction (overall comparison: uncorrected); grey: not significant.

## Fig05_localization

Overlap (Rel-90 IoU) between the explanation and the radiologist's bounding box for correctly classified positive patients (true positives), on the clean image and under each perturbation level. Shaded bands: 95 % patient-bootstrap confidence intervals.

## Fig06_background_activation

Background activation of positive patients: share of pixels outside the bounding box with normalised activation >= 0.5, on the clean image and under each perturbation level. This absolute-threshold measure replaces the percentile-based false activation rate of the original submission. Shaded bands: 95 % CIs.

## Fig07_validity

XFI follows the degradation of the input and the change in localization. (a) Mean XFI by bin of input PSNR (clean against perturbed image, 3 dB bins, all conditions pooled); Spearman correlation between XFI and input PSNR ranges from -0.61 to -0.89. (b) Mean absolute change in Rel-90 IoU by XFI decile, positive patients; Spearman correlation 0.41 to 0.61. Colour: explanation method; line style: classifier.

## Fig08_baselines

Patient-level XFI (mean over the three perturbation types) per severity level for Grad-CAM and Eigen-CAM, against baselines (horizontal lines): explanations of two different patients (cross-patient), explanations of a copy of the classifier whose last residual block and output layer were re-initialised (randomised model), two independent smooth random maps (7 x 7 upsampled) and two independent pixel-noise maps. Boxes: interquartile range and median; whiskers: 1.5 IQR.

## Fig09_prediction_flips

(a) Share of patients whose predicted class changes under each perturbation condition, per classifier (shaded: 95 % CI). Under Gaussian noise L2 and L3 every classifier assigns almost every image to one class (negative for Models A and B, positive for Models A-rep and A-norm), so the flip rate equals the share of patients assigned to the other class on the clean image. (b) Mean XFI of perturbed images with (filled) and without (open) a change of the predicted class, all conditions pooled.

## Fig10_examples_modelA

Representative explanations of Model A for true-positive patients (selection rule: Grad-CAM XFI under Gaussian noise L3 closest to the median of the stored true-positive sample). Columns: input with the radiologist's bounding box (blue), then Grad-CAM and Eigen-CAM on the clean image, under Gaussian noise L3 and under salt-and-pepper noise L3, overlaid on the corresponding input. Arrows: shift of the activation centre of mass from the clean to the perturbed map; numbers: XFI.

## Fig11_com_drift

Shift of the activation centre of mass between the clean and the perturbed explanation (pixels on the 224 x 224 grid), per perturbation type and level (three boxes per method, L1 to L3 from left to right). Boxes: interquartile range and median; whiskers: 1.5 IQR.

## FigS1_kinetics_all_models

Mean XFI against perturbation severity for Grad-CAM and Eigen-CAM. Rows: all four classifiers; columns: Gaussian noise (sigma = 0.1, 0.2, 0.3), Gaussian blur (kernel 3, 5, 7) and salt-and-pepper noise (1, 2, 3 % of pixels). Shaded bands: 95 % patient-bootstrap confidence intervals. Dashed lines: cross-patient XFI of the same method and classifier, i.e. the typical dissimilarity between the clean explanations of two different patients.

## FigS2_localization_all_models

Overlap (Rel-90 IoU) between the explanation and the radiologist's bounding box for correctly classified positive patients (true positives), on the clean image and under each perturbation level. Shaded bands: 95 % patient-bootstrap confidence intervals. All four classifiers.

## FigS3_background_activation_all_models

Background activation of positive patients: share of pixels outside the bounding box with normalised activation >= 0.5, on the clean image and under each perturbation level. This absolute-threshold measure replaces the percentile-based false activation rate of the original submission. Shaded bands: 95 % CIs. All four classifiers.

## FigS4_eigencam_sign

Eigen-CAM sign behaviour. (a) Share of perturbed images whose signed first-principal-component projection points opposite to that of the clean image. Line style: classifier. (b-e) Mean XFI of Eigen-CAM without and with sign correction (the projection is negated when its minimum exceeds its maximum in magnitude), three levels per perturbation type.

## FigS5_examples_modelB

Representative explanations of Model B for true-positive patients (selection rule: Grad-CAM XFI under Gaussian noise L3 closest to the median of the stored true-positive sample). Columns: input with the radiologist's bounding box (blue), then Grad-CAM and Eigen-CAM on the clean image, under Gaussian noise L3 and under salt-and-pepper noise L3, overlaid on the corresponding input. Arrows: shift of the activation centre of mass from the clean to the perturbed map; numbers: XFI.

