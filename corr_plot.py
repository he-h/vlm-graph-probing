import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# ---------- Config ----------
BASE = Path("results/modality_correlation")
TASK = "tdiuc_color"          # or "clevr_counting"
FIGSIZE = (10.5, 4.2)         # (7, 3.5) for single-column
DPI = 300

colors = {"vv": "#3498db", "vt": "#e74c3c", "tt": "#2ecc71"}
lw = 1.6
alpha_fill = 0.18

models = [
    ("InternVL3-1B",  f"modality_corr_InternVL3-1B_{TASK}.npy"),
    ("Qwen2.5-VL-3B", f"modality_corr_Qwen2.5-VL-3B_{TASK}.npy"),
    ("LLaVA-1.5-7B",  f"modality_corr_LLaVA-1.5-7B_{TASK}.npy"),
    ("InternVL3-2B",  f"modality_corr_InternVL3-2B_{TASK}.npy"),
    ("Qwen2.5-VL-7B", f"modality_corr_Qwen2.5-VL-7B_{TASK}.npy"),
    ("LLaVA-1.5-13B", f"modality_corr_LLaVA-1.5-13B_{TASK}.npy"),
]

def blob_has_only_finite(b):
    arrs = [
        np.asarray(b["vv_mean"]), np.asarray(b["vv_std"]),
        np.asarray(b["vt_mean"]), np.asarray(b["vt_std"]),
        np.asarray(b["tt_mean"]), np.asarray(b["tt_std"]),
    ]
    return all(np.isfinite(a).all() and a.size > 0 for a in arrs)

# ---------- Load, validate, and collect bounds from valid panels ----------
loaded, valid_mask, bounds = [], [], []
for name, fname in models:
    blob = np.load(BASE / fname, allow_pickle=True).item()
    is_valid = blob_has_only_finite(blob)
    valid_mask.append(is_valid)
    loaded.append((name, blob))
    if is_valid:
        for key in ("vv", "vt", "tt"):
            m = np.asarray(blob[f"{key}_mean"])
            s = np.asarray(blob[f"{key}_std"])
            bounds.extend([np.min(m - s), np.max(m + s)])

if not bounds:
    raise RuntimeError("All panels invalid (contain NaN/Inf). Nothing to plot.")

ymin, ymax = float(np.min(bounds)), float(np.max(bounds))
pad = 0.03 * (ymax - ymin if ymax > ymin else 1.0)
ymin, ymax = ymin - pad, ymax + pad

# ---------- Matplotlib pub settings ----------
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 10, "savefig.bbox": "tight",
})

fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, dpi=DPI, constrained_layout=False)

def plot_panel(ax, name, blob, show_xlabel: bool, show_ylabel: bool):
    vv_m, vv_s = np.asarray(blob["vv_mean"]), np.asarray(blob["vv_std"])
    vt_m, vt_s = np.asarray(blob["vt_mean"]), np.asarray(blob["vt_std"])
    tt_m, tt_s = np.asarray(blob["tt_mean"]), np.asarray(blob["tt_std"])
    L = len(vv_m); x = np.arange(L)

    # mean ± std bands
    ax.plot(x, vv_m, lw=lw, label="Vision–Vision", color=colors["vv"])
    ax.fill_between(x, vv_m - vv_s, vv_m + vv_s, alpha=alpha_fill, edgecolor="none", color=colors["vv"])
    ax.plot(x, vt_m, lw=lw, label="Vision–Text", color=colors["vt"])
    ax.fill_between(x, vt_m - vt_s, vt_m + vt_s, alpha=alpha_fill, edgecolor="none", color=colors["vt"])
    ax.plot(x, tt_m, lw=lw, label="Text–Text", color=colors["tt"])
    ax.fill_between(x, tt_m - tt_s, tt_m + tt_s, alpha=alpha_fill, edgecolor="none", color=colors["tt"])

    ax.set_title(name)
    if show_xlabel:
        ax.set_xlabel("Layer")
    else:
        ax.set_xlabel("")  # remove
        ax.tick_params(axis='x', labelbottom=False)  # hide tick labels for top row

    if show_ylabel:
        ax.set_ylabel("Correlation")
    else:
        ax.set_ylabel("")  # remove

    ax.set_xlim(0, L - 1)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.3)

# Draw panels (skip invalid ones) with conditional labels
for idx, ax in enumerate(axes.flatten()):
    name, blob = loaded[idx]
    row, col = divmod(idx, 3)
    show_xlabel = (row == 1)          # only bottom row
    show_ylabel = (col == 0)          # only first column
    if valid_mask[idx]:
        plot_panel(ax, name, blob, show_xlabel, show_ylabel)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{name}\nSkipped (NaN/Inf)", ha="center", va="center", fontsize=9)

# Panel labels (a)-(f)
for i, ax in enumerate(axes.flatten()):
    if ax.axison:
        ax.text(0.02, 0.95, f"({chr(97+i)})", transform=ax.transAxes,
                ha="left", va="top", weight="bold")

# Shared legend with full names
handles = [
    Line2D([0],[0], color=colors["vv"], lw=lw, label="Vision–Vision"),
    Line2D([0],[0], color=colors["vt"], lw=lw, label="Vision–Text"),
    Line2D([0],[0], color=colors["tt"], lw=lw, label="Text–Text"),
]
fig.legend(handles=handles, labels=[h.get_label() for h in handles],
           ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)

fig.subplots_adjust(wspace=0.28, hspace=0.35, top=0.88)

out_base = f"modality_corr_{TASK}_6panels"
fig.savefig(out_base + ".pdf")
print("Saved:", out_base + ".pdf")
plt.show()
