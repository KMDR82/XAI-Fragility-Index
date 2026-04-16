import os
import gc
import cv2
import glob
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pydicom
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, wilcoxon
from skimage.metrics import structural_similarity as ssim
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

try:
    from scipy.integrate import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

# ====================== REPRODUCIBILITY ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("======================================================")
print("XAI FRAGILITY INDEX (XFI) FRAMEWORK - FULL EXPERIMENT")
print("======================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# ==========================================
# 0. DIRECTORY DISCOVERY
# ==========================================
print("\nScanning dataset directories...")
csv_search = glob.glob('/kaggle/input/**/stage_2_train_labels.csv', recursive=True)
if not csv_search:
    raise ValueError("ERROR: 'stage_2_train_labels.csv' not found!")
CSV_PATH = csv_search[0]

IMAGE_DIR = None
for ext in ['*.dcm', '*.png', '*.jpg']:
    search_res = glob.glob(f'/kaggle/input/**/{ext}', recursive=True)
    if search_res:
        IMAGE_DIR = os.path.dirname(search_res[0])
        break

if not IMAGE_DIR:
    raise ValueError("ERROR: Image files not found!")

print(f"Image Directory: {IMAGE_DIR}")

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("Preparing dataset...")
df_labels = pd.read_csv(CSV_PATH)
all_patients = df_labels['patientId'].unique()

existing_files = set(os.listdir(IMAGE_DIR))
valid_patients = [pid for pid in all_patients 
                  if f"{pid}.dcm" in existing_files or 
                     f"{pid}.png" in existing_files or 
                     f"{pid}.jpg" in existing_files]

print(f" - Total Patients (CSV): {len(all_patients)}")
print(f" - Valid Patients (Files present): {len(valid_patients)}")

train_ids, test_ids = train_test_split(valid_patients, test_size=0.2, random_state=SEED)

def get_bounding_boxes(patient_id):
    boxes = df_labels[df_labels['patientId'] == patient_id]
    bbox_list = []
    for _, row in boxes.iterrows():
        if row['Target'] == 1 and not pd.isna(row['x']):
            bbox_list.append([row['x'], row['y'], row['width'], row['height']])
    return bbox_list

class RSNADataset(Dataset):
    def __init__(self, patient_ids, df_labels, img_dir, transform=None):
        self.patient_ids = patient_ids
        self.df_labels = df_labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        row = self.df_labels[self.df_labels['patientId'] == pid].iloc[0]
        target = row['Target']

        for ext in ['.dcm', '.png', '.jpg']:
            path = os.path.join(self.img_dir, f"{pid}{ext}")
            if os.path.exists(path):
                if ext == '.dcm':
                    dicom = pydicom.dcmread(path)
                    image = dicom.pixel_array
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                break

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32), pid

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = RSNADataset(train_ids, df_labels, IMAGE_DIR, transform=train_transform)
test_dataset = RSNADataset(test_ids, df_labels, IMAGE_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 2. MODEL TRAINING
# ==========================================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

MODEL_SAVE_PATH = '/kaggle/working/best_pneumonia_model.pth'
best_loss = float('inf')

print("\nStarting model training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for i, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0 and i > 0:
            print(f" [Epoch {epoch+1}/{EPOCHS}, Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Completed | Avg Loss: {epoch_loss:.4f} | Time: {time.time()-start_time:.1f}s")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   Best model saved (Loss: {best_loss:.4f})")

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
print(f"\nTraining completed. Model loaded: {MODEL_SAVE_PATH}")

# ==========================================
# 3. XAI & XFI FRAMEWORK
# ==========================================
target_layers = [model.layer4[-1]]

class XAIGenerator:
    def __init__(self, model, layers):
        self.grad_cam = GradCAM(model=model, target_layers=layers)
        self.eigen_cam = EigenCAM(model=model, target_layers=layers)
        
    def generate_heatmap(self, method, input_tensor, targets):
        if method == 'GradCAM':
            return self.grad_cam(input_tensor=input_tensor, targets=targets)[0]
        elif method == 'EigenCAM':
            return self.eigen_cam(input_tensor=input_tensor, targets=targets)[0]

xai_gen = XAIGenerator(model, target_layers)

def apply_salt_pepper_v11(img, amount=0.02):
    res = img.copy()
    h, w = img.shape[:2]
    salt_val = np.percentile(img, 99)
    pepper_val = np.percentile(img, 1)
    num_pixels = int(amount * h * w)
    res[np.random.randint(0, h, num_pixels//2), np.random.randint(0, w, num_pixels//2)] = salt_val
    res[np.random.randint(0, h, num_pixels//2), np.random.randint(0, w, num_pixels//2)] = pepper_val
    return res

def tensor_to_np(tensor):
    np_img = tensor.detach().cpu().squeeze().numpy()
    if np_img.ndim == 3:
        np_img = np.transpose(np_img, (1, 2, 0))
    return np_img

def np_to_tensor(np_img, device):
    if np_img.ndim == 3:
        np_img = np.transpose(np_img, (2, 0, 1))
    return torch.tensor(np_img, dtype=torch.float32).unsqueeze(0).to(device)

class XAI_Robustness_Framework_V11:
    def __init__(self, rel_thresholds=[0.80, 0.90], abs_thresholds=[0.5, 0.7]):
        self.rel_thresholds = rel_thresholds
        self.abs_thresholds = abs_thresholds

    def normalize_hm(self, hm):
        diff = hm.max() - hm.min()
        return (hm - hm.min()) / (diff + 1e-8)

    def evaluate_sample(self, raw_clean, raw_noisy, bboxes, label, orig_shape=(1024, 1024)):
        c_hm = self.normalize_hm(raw_clean)
        n_hm = self.normalize_hm(raw_noisy)

        # SSIM
        if np.std(c_hm) < 1e-8 or np.std(n_hm) < 1e-8:
            s_sim = 0.0
        else:
            s_sim = ssim(c_hm, n_hm, data_range=1.0)

        # Correlations
        c_f, n_f = c_hm.flatten(), n_hm.flatten()
        p_corr, _ = pearsonr(c_f, n_f)
        s_corr, _ = spearmanr(c_f, n_f)

        p_corr_n = (p_corr + 1) / 2 if not np.isnan(p_corr) else 0.5
        s_corr_n = (s_corr + 1) / 2 if not np.isnan(s_corr) else 0.5

        robustness_score = (s_sim + p_corr_n + s_corr_n) / 3
        xfi = 1.0 - robustness_score                     # XFI calculation

        results = {
            'ssim': s_sim,
            'robustness_score': robustness_score,   
            'xfi': xfi,                             
            'p_corr': p_corr,
            's_corr': s_corr
        }

        # False Activation and IoU Computations
        if label == 0:
            results['false_activation'] = np.sum(n_hm > np.percentile(n_hm, 90)) / n_hm.size
        else:
            target_h, target_w = c_hm.shape
            orig_h, orig_w = orig_shape
            gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            for box in bboxes:
                x1 = np.clip(int(box[0] * target_w / orig_w), 0, target_w)
                y1 = np.clip(int(box[1] * target_h / orig_h), 0, target_h)
                x2 = np.clip(int((box[0] + box[2]) * target_w / orig_w), 0, target_w)
                y2 = np.clip(int((box[1] + box[3]) * target_h / orig_h), 0, target_h)
                gt_mask[y1:y2, x1:x2] = 1

            t90 = np.percentile(n_hm, 90)
            results['false_activation'] = np.sum((n_hm > t90) & (gt_mask == 0)) / (np.sum(gt_mask == 0) + 1e-8)

            for t in self.rel_thresholds:
                mask_c_rel = (c_hm >= np.percentile(c_hm, t*100)).astype(np.uint8)
                mask_n_rel = (n_hm >= np.percentile(n_hm, t*100)).astype(np.uint8)
                results[f'iou_c_rel_{int(t*100)}'] = np.sum(mask_c_rel & gt_mask) / (np.sum(mask_c_rel | gt_mask) + 1e-8)
                results[f'iou_n_rel_{int(t*100)}'] = np.sum(mask_n_rel & gt_mask) / (np.sum(mask_n_rel | gt_mask) + 1e-8)

            for t in self.abs_thresholds:
                mask_c_abs = (c_hm >= t).astype(np.uint8)
                mask_n_abs = (n_hm >= t).astype(np.uint8)
                results[f'iou_c_abs_{int(t*10)}'] = np.sum(mask_c_abs & gt_mask) / (np.sum(mask_c_abs | gt_mask) + 1e-8)
                results[f'iou_n_abs_{int(t*10)}'] = np.sum(mask_n_abs & gt_mask) / (np.sum(mask_n_abs | gt_mask) + 1e-8)

        return results

# ==========================================
# 4. REPORTING FUNCTION
# ==========================================
def generate_q1_report(df):
    print("\n" + "="*80)
    print("XAI FRAGILITY INDEX (XFI) FRAMEWORK - FINAL SCIENTIFIC REPORT")
    print("="*80)
    
    print("\n1. PAIRED STATISTICAL SIGNIFICANCE (GradCAM vs EigenCAM):")
    for metric in ['ssim', 'xfi']:
        pivot = df.pivot_table(index=['Patient', 'Level', 'Type'], columns='Method', values=metric).dropna()
        if not pivot.empty:
            diff = pivot['GradCAM'] - pivot['EigenCAM']
            if np.allclose(diff, 0): continue
            stat, p = wilcoxon(pivot['GradCAM'], pivot['EigenCAM'])
            n = len(pivot)
            mu_w = n * (n + 1) / 4
            sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z_stat = (stat - mu_w) / sigma_w
            r = abs(z_stat) / np.sqrt(n)
            print(f" - {metric:16} | p-value: {p:.4e} | Z: {z_stat:.4f} | Effect Size (r): {r:.4f}")

    print("\n2. DEGRADATION KINETICS (AUC & Slope) [95% CI]:")
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        x = sorted(sub['Level'].unique())
        if len(x) > 1:
            y = sub.groupby('Level')['xfi'].mean().loc[x].values
            norm_auc = trapezoid(y, x) / (x[-1] - x[0])
            slope = np.polyfit(x, y, 1)[0]
            mean_score = sub['xfi'].mean()
            std_score = sub['xfi'].std()
            ci_margin = 1.96 * (std_score / np.sqrt(len(sub)))
            print(f" - {method:18} | Mean XFI: {mean_score:.4f} (+/-{ci_margin:.4f}) | AUC: {norm_auc:.4f} | Slope: {slope:.4f}")

    print("\n3. LOCALIZATION SHIFT (IoU Rel-90 Drop) & SPECIFICITY:")
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        fa = sub['false_activation'].mean() if 'false_activation' in sub.columns else 0
        if 'iou_c_rel_90' in sub.columns:
            c_iou = sub['iou_c_rel_90'].mean()
            n_iou = sub['iou_n_rel_90'].mean()
            drop = ((c_iou - n_iou) / (c_iou + 1e-8)) * 100
            print(f" - {method:18} | Clean IoU: {c_iou:.4f} -> Noisy IoU: {n_iou:.4f} (Drop: {drop:5.1f}%) | False Act: {fa:.4f}")
            
    print("\n" + "="*80)

# ==========================================
# 5. TESTING PHASE
# ==========================================
print("\nInitiating XAI Fragility Index analysis (Full test set)...\n")

results_list = []
evaluator = XAI_Robustness_Framework_V11()
processed = 0
total_patients = len(test_loader)

for images, labels, pids in test_loader:
    clean_tensor = images.to(device).requires_grad_(True)
    label = labels.item()
    pid = pids[0]
    bboxes = get_bounding_boxes(pid)
    clean_np = tensor_to_np(clean_tensor)

    with torch.no_grad():
        logits = model(clean_tensor)
        prob = torch.sigmoid(logits).item()
        target_class = 1 if prob > 0.5 else 0
        targets = [BinaryClassifierOutputTarget(target_class)]

    methods = ['GradCAM', 'EigenCAM', 'Random_Baseline', 'Uniform_Baseline']

    for m_name in methods:
        if m_name == 'Random_Baseline':
            raw_n = np.random.rand(224, 224)
            raw_c = (raw_n - raw_n.min()) / (raw_n.max() - raw_n.min() + 1e-8)
        elif m_name == 'Uniform_Baseline':
            raw_c = np.ones((224, 224)) * 0.5
        else:
            model.zero_grad()
            raw_c = xai_gen.generate_heatmap(m_name, clean_tensor, targets)

        for p_type in ['Gaussian', 'Blur', 'SaltPepper']:
            for level in [1, 2, 3]:
                if p_type == 'Gaussian':
                    std = level * 0.1
                    noise = np.random.normal(0, std, clean_np.shape)
                    noisy_np = np.clip(clean_np + noise, 0, 1).astype(np.float32)
                elif p_type == 'Blur':
                    k = int(level * 2 + 1)
                    noisy_np = cv2.GaussianBlur(clean_np, (k, k), 0)
                else:
                    noisy_np = apply_salt_pepper_v11(clean_np, amount=0.01 * level)

                noisy_tensor = np_to_tensor(noisy_np, device).requires_grad_(True)

                if m_name in ['Random_Baseline', 'Uniform_Baseline']:
                    raw_n_hm = raw_c.copy()
                else:
                    model.zero_grad()
                    raw_n_hm = xai_gen.generate_heatmap(m_name, noisy_tensor, targets)

                metrics = evaluator.evaluate_sample(raw_c, raw_n_hm, bboxes, label)

                res = {
                    'Patient': pid,
                    'Label': label,
                    'Model_Prob': prob,
                    'Method': m_name,
                    'Type': p_type,
                    'Level': level,
                    **metrics
                }
                results_list.append(res)

                del noisy_tensor
                if m_name not in ['Random_Baseline', 'Uniform_Baseline']:
                    del raw_n_hm

        if m_name not in ['Random_Baseline', 'Uniform_Baseline']:
            del raw_c

    processed += 1
    if processed % 200 == 0 or processed == total_patients:
        print(f"Progress: {processed}/{total_patients} patients processed...")
        torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# SAVING RESULTS
# ==========================================
df_final = pd.DataFrame(results_list)
csv_name = 'xfi_full_experiment_results.csv'
df_final.to_csv(csv_name, index=False)

print(f"\nEXPERIMENT COMPLETED! Results saved to '{csv_name}'.")
print(f"   -> XFI column added. High XFI = More fragile heatmap")

generate_q1_report(df_final)