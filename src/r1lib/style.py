"""Figure conventions (D-20).

- no title inside the figure (caption carries it)
- panel labels (a) (b) (c) centred below the x-axis: ax.text(0.5, -0.18, ..., transform=ax.transAxes)
- every figure saved as PNG and PDF at 300 dpi, and the data behind it saved as CSV
- categorical colours validated for colour-vision deficiency (dataviz validator, light mode):
  Eigen-CAM blue #2a78d6, Grad-CAM orange #eb6834 (worst CVD dE 24.7, all checks pass)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

METHOD_COLORS = {"GradCAM": "#eb6834", "EigenCAM": "#2a78d6"}
METHOD_LABELS = {"GradCAM": "Grad-CAM", "EigenCAM": "Eigen-CAM",
                 "Random_Baseline": "Random baseline", "Uniform_Baseline": "Uniform baseline"}
METHOD_MARKERS = {"GradCAM": "o", "EigenCAM": "s"}
METHODS = ["GradCAM", "EigenCAM"]
TYPES = ["Gaussian", "Blur", "SaltPepper"]
TYPE_LABELS = {"Gaussian": "Gaussian noise", "Blur": "Gaussian blur", "SaltPepper": "Salt-and-pepper"}
NEUTRAL = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", ["#2a78d6", "#f0efec", "#eb6834"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f3f7fd", "#2a78d6", "#0d3a73"])

PANEL_Y = -0.18


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, letter, y=None):
    """(a), (b), ... centred below the x-axis at y = -0.18 (axes fraction). If the x tick labels or the
    x-axis label reach lower than that, save_figure() moves the label just below them, keeping all labels
    of one row of panels on the same line. Pass y to fix the position manually."""
    t = ax.text(0.5, PANEL_Y if y is None else y, f"({letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold", color=INK)
    ax._r1_panel = (t, y is not None)


def _place_panel_labels(fig, gap_pt=3.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rows = {}
    for ax in fig.axes:
        if not hasattr(ax, "_r1_panel") or ax._r1_panel[1]:
            continue
        t, _ = ax._r1_panel
        bottom_disp = ax.xaxis.get_tightbbox(r).y0 - gap_pt * fig.dpi / 72
        default_disp = ax.transAxes.transform((0, PANEL_Y))[1]
        rows.setdefault(round(ax.get_position().y0, 3), []).append((ax, t, min(bottom_disp, default_disp)))
    for items in rows.values():
        ydisp = min(i[2] for i in items)
        for ax, t, _ in items:
            t.set_y(ax.transAxes.inverted().transform((0, ydisp))[1])


def save_figure(fig, stem, data, outdir):
    """Save fig as <stem>.png and <stem>.pdf (300 dpi) and its data as <stem>.csv
    (or <stem>__<key>.csv for a dict of DataFrames). Returns the list of written paths."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    _place_panel_labels(fig)
    paths = []
    for ext in ("png", "pdf"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
        paths.append(p)
    if isinstance(data, dict):
        for k, df in data.items():
            p = outdir / f"{stem}__{k}.csv"; df.to_csv(p, index=False); paths.append(p)
    else:
        p = outdir / f"{stem}.csv"; data.to_csv(p, index=False); paths.append(p)
    plt.close(fig)
    return paths
