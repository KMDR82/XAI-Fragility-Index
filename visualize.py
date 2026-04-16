import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Attempt to load from Kaggle path first, fallback to local directory
PRIMARY_CSV_PATH = '/kaggle/input/datasets/aakkaya/full_experiment_results_xfi.csv'
FALLBACK_CSV_PATH = 'full_experiment_results_xfi.csv'
OUTPUT_DIR = 'figures'

# Academic Journal Styling Standards
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.dpi': 300,             
    'font.family': 'sans-serif',   
    'font.size': 12,               
    'axes.labelsize': 14,          
    'axes.titlesize': 14,          
    'legend.fontsize': 12,         
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.5,         
    'figure.autolayout': True
})

# Standardized Color Palette (Colorblind-friendly red/blue contrast)
METHOD_PALETTE = {
    'GradCAM': '#e74c3c',  # Red
    'EigenCAM': '#3498db'  # Blue
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    """Loads and filters dataset, keeping only target XAI methods."""
    if os.path.exists(PRIMARY_CSV_PATH):
        path = PRIMARY_CSV_PATH
    elif os.path.exists(FALLBACK_CSV_PATH):
        path = FALLBACK_CSV_PATH
    else:
        raise FileNotFoundError(f"Dataset not found at {PRIMARY_CSV_PATH} or {FALLBACK_CSV_PATH}")
    
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    
    # Critical Filter: Keep only GradCAM and EigenCAM (Exclude baselines)
    df_xai = df[df['Method'].isin(['GradCAM', 'EigenCAM'])].copy()
    return df_xai

# ==========================================
# FIGURE GENERATION FUNCTIONS
# ==========================================

def plot_figure_2_kinetics(df):
    """Figure 2: XFI trends across noise severity levels (Fragility Kinetics)."""
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, x='Level', y='xfi', hue='Method', 
        palette=METHOD_PALETTE, marker='o', markersize=8, 
        linewidth=2.5, errorbar='sd'
    )

    plt.xlabel('Perturbation Level (Increasing Noise)', fontweight='bold')
    plt.ylabel('XAI Fragility Index (XFI)', fontweight='bold')
    plt.xticks([1, 2, 3], ['Level 1\n(Mild)', 'Level 2\n(Moderate)', 'Level 3\n(Severe)'])
    plt.ylim(0.0, 0.8)
    plt.legend(title='Explanation Method', loc='upper left')
    plt.title('') # Removed title as per reviewer guidelines
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure02_XFI_Kinetics.png'), dpi=300)
    plt.close()
    print(" - Figure 2 (Fragility Kinetics) generated.")

def plot_figure_3_xfi_violin(df):
    """Figure 3: Statistical distribution of XFI scores across degradation levels."""
    # Speed optimization: Sample down if dataset is extremely large
    df_sampled = df.sample(n=min(50000, len(df)), random_state=42)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df_sampled, x='Level', y='xfi', hue='Method',
        split=True, inner='quartile', palette=METHOD_PALETTE, linewidth=1.5
    )

    plt.xlabel('Perturbation Level', fontweight='bold')
    plt.ylabel('XAI Fragility Index (XFI)', fontweight='bold')
    plt.xticks([0, 1, 2], ['Level 1\n(Mild)', 'Level 2\n(Moderate)', 'Level 3\n(Severe)'])
    plt.legend(title='XAI Method', loc='upper left')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure03_XFI_Violin.png'), dpi=300)
    plt.close()
    print(" - Figure 3 (XFI Violin Distribution) generated.")

def plot_figure_4_xfi_artifact_type(df):
    """Figure 4: Statistical distribution of XFI across different artifact types."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df, x='Type', y='xfi', hue='Method',
        palette=METHOD_PALETTE, width=0.6, fliersize=3
    )

    plt.xlabel('Artifact Type', fontweight='bold')
    plt.ylabel('XAI Fragility Index (XFI)', fontweight='bold')
    plt.legend(title='XAI Method', loc='upper right')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure04_XFI_ArtifactType_Boxplot.png'), dpi=300)
    plt.close()
    print(" - Figure 4 (XFI by Artifact Type) generated.")

def plot_figure_5_delta_iou(df):
    """Figure 5: Absolute localization shifts (Delta IoU) by noise severity."""
    # Delta IoU = Clean - Noisy
    df['delta_iou'] = df['iou_c_rel_80'] - df['iou_n_rel_80']

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, x='Level', y='delta_iou', hue='Method', 
        palette=METHOD_PALETTE, showfliers=False
    )
    
    plt.xlabel('Noise Level', fontweight='bold')
    plt.ylabel(r'$\Delta$ IoU (Clean - Noisy)', fontweight='bold')
    plt.xticks([0, 1, 2], ['L1', 'L2', 'L3'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure05_IoU_Stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Figure 5 (Delta IoU Stability) generated.")

def plot_figure_6_level3_clinical_errors(df):
    """Figure 6: Clinical error rates under the most severe degradation (Level-3)."""
    df_extreme = df[df['Level'] == 3].copy()
    
    # Calculate False Activation Rate (FAR) percentage
    df_extreme['false_act_pct'] = df_extreme['false_activation'] * 100
    
    # Calculate IoU Degradation percentage relative to clean image
    df_extreme['iou_drop'] = np.where(
        df_extreme['iou_c_rel_90'] > 0, 
        ((df_extreme['iou_c_rel_90'] - df_extreme['iou_n_rel_90']) / df_extreme['iou_c_rel_90']) * 100, 
        0
    )
    df_extreme['iou_drop'] = df_extreme['iou_drop'].clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Background Leakage
    sns.barplot(
        data=df_extreme, x='Type', y='false_act_pct', hue='Method', 
        palette=METHOD_PALETTE, capsize=.1, err_kws={'linewidth': 1.5}, ax=axes[0]
    )
    axes[0].set_title('A) Background Leakage Under $L_3$ Noise (Lower is Better)', fontweight='bold')
    axes[0].set_ylabel('False Activation Rate (%)', fontweight='bold')
    axes[0].set_xlabel('Perturbation Type', fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].set_ylim(0, max(df_extreme['false_act_pct']) * 1.2)

    # Panel B: IoU Degradation
    sns.barplot(
        data=df_extreme, x='Type', y='iou_drop', hue='Method', 
        palette=METHOD_PALETTE, capsize=.1, err_kws={'linewidth': 1.5}, ax=axes[1]
    )
    axes[1].set_title('B) Target Localization Drop Under $L_3$ Noise (Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('IoU Degradation (%)', fontweight='bold')
    axes[1].set_xlabel('Perturbation Type', fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure06_Clinical_Relevance_Barplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Figure 6 (Level-3 Clinical Errors) generated.")

def plot_figure_7_false_activation_violin(df):
    """Figure 7: False Activation Rate (FAR) distributions across noise levels."""
    plt.figure(figsize=(9, 6))
    sns.violinplot(
        data=df, x='Level', y='false_activation', hue='Method', 
        palette=METHOD_PALETTE, split=True, inner="quartile", linewidth=1.5
    )
    
    plt.xlabel('Noise Level', fontweight='bold')
    plt.ylabel('False Activation Rate', fontweight='bold')
    plt.xticks([0, 1, 2], ['L1', 'L2', 'L3'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure07_False_Activation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Figure 7 (False Activation Analysis) generated.")

def plot_figure_8_correlation_matrix(df):
    """Figure 8: Pearson correlation matrix between XFI and other evaluation metrics."""
    metrics = ['ssim', 'xfi', 'p_corr', 's_corr', 'false_activation', 'iou_c_rel_80']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    corr_matrix = df[available_metrics].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, 
        cbar_kws={'label': 'Correlation Coefficient'}, square=True
    )

    labels = ['SSIM', 'XFI', 'Pearson', 'Spearman', 'False Act.', 'IoU'][:len(available_metrics)]
    plt.xticks(ticks=np.arange(len(labels))+0.5, labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels))+0.5, labels=labels, rotation=0)

    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure08_Metric_Correlation_XFI.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Figure 8 (Metric Correlation Heatmap) generated.")

def plot_figure_9_ssim_vs_xfi(df):
    """Figure 9: Scatter plot illustrating the relationship between SSIM and XFI."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='ssim', y='xfi', hue='Method', 
        palette=METHOD_PALETTE, alpha=0.5, edgecolor=None, s=50
    )
    
    plt.xlabel('Structural Similarity Index (SSIM)', fontweight='bold')
    plt.ylabel('XAI Fragility Index (XFI)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure09_SSIM_vs_XFI.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Figure 9 (SSIM vs XFI Scatter Plot) generated.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing Q1 Publication Visualization Pipeline...")
    ensure_dir(OUTPUT_DIR)
    
    try:
        df_results = load_data()
        
        print(f"\nGenerating Figures ({len(df_results)} rows)...")
        plot_figure_2_kinetics(df_results)
        plot_figure_3_xfi_violin(df_results)
        plot_figure_4_xfi_artifact_type(df_results)
        plot_figure_5_delta_iou(df_results)
        plot_figure_6_level3_clinical_errors(df_results)
        plot_figure_7_false_activation_violin(df_results)
        plot_figure_8_correlation_matrix(df_results)
        plot_figure_9_ssim_vs_xfi(df_results)
        
        print(f"\nExecution Complete! All high-resolution figures saved to '{OUTPUT_DIR}' directory.")
    
    except Exception as e:
        print(f"\nError encountered: {e}")
