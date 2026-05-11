import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
from pathlib import Path
from settings import (
    EMBEDDING_OUTPUT_DIR,
    GENERATED_OUTPUT_PATH,
    EMBEDDING_METADATA_FILENAME,
    UMAP_PROJECTIONS_FILENAME,
    OPTICS_LABELS_FILENAME,
    OPTICS_REACHABILITY_FILENAME,
    OPTICS_ORDERINGS_FILENAME,
    OPTICS_METRICS_FILENAME,
)

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# --- Load all precomputed data from precompute_umap.py ---

# projections: [num_tokens, n_prompts, num_return_per_prompt, 2], float32
projections = np.load(output_dir / UMAP_PROJECTIONS_FILENAME)
num_tokens, n_prompts, num_return_per_prompt, _ = projections.shape

# optics_labels:       [num_tokens, n_prompts, num_return_per_prompt], int32
# optics_reachability: [num_tokens, n_prompts, num_return_per_prompt], float32 (ordered, inf->1.0)
# optics_orderings:    [num_tokens, n_prompts, num_return_per_prompt], int64
optics_labels       = np.load(output_dir / OPTICS_LABELS_FILENAME)
optics_reachability = np.load(output_dir / OPTICS_REACHABILITY_FILENAME)
optics_orderings    = np.load(output_dir / OPTICS_ORDERINGS_FILENAME)

# optics_metrics keys — each shape [num_tokens, n_prompts]
optics_metrics = np.load(output_dir / OPTICS_METRICS_FILENAME)
metrics_cluster_counts   = optics_metrics["cluster_counts"]    # int32
metrics_noise_counts     = optics_metrics["noise_counts"]      # int32
metrics_core_counts      = optics_metrics["core_counts"]       # int32
metrics_border_counts    = optics_metrics["border_counts"]     # int32
metrics_core_homogeneity = optics_metrics["core_homogeneity"]  # float32

# metadata: list of dicts with prompt_text, sequence_index
with open(output_dir / EMBEDDING_METADATA_FILENAME, "r", encoding="utf-8") as f:
    metadata = json.load(f)
prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))

# entropies: [num_prompts, num_return, max_length], float32 — averaged over returns for bottom panel
entropy_data = np.load(GENERATED_OUTPUT_PATH, allow_pickle=False)
entropy_prompt_texts = [str(p) for p in entropy_data["prompt_texts"]]
entropy_by_prompt = entropy_data["entropies"].mean(axis=1)   # [num_prompts, max_length]
entropy_tokens = np.arange(1, entropy_by_prompt.shape[1] + 1)

# --- Derive display arrays at load time (pure numpy, no sklearn) ---

# flat_projections[t]: [num_sentences, 2] — all prompts concatenated for single scatter
# flat_prompt_ids:     [num_sentences]    — prompt index per point, same order as flat_projections
flat_projections = np.concatenate([projections[:, p, :, :] for p in range(n_prompts)], axis=1)
# flat_projections: [num_tokens, num_sentences, 2]
flat_prompt_ids = np.concatenate([np.full(num_return_per_prompt, p) for p in range(n_prompts)])
# flat_prompt_ids: [num_sentences]

# scaled_reachability[t, p]: [num_return_per_prompt] — log1p of ordered reachability
def scale_reachability_for_plot(vals):
    return np.log1p(vals)

scaled_reachability = scale_reachability_for_plot(optics_reachability)
# scaled_reachability: [num_tokens, n_prompts, num_return_per_prompt], float32

global_reach_max = float(np.max(optics_reachability))
if not np.isfinite(global_reach_max) or global_reach_max <= 0:
    global_reach_max = 1.0
global_prompt_reach_max = scale_reachability_for_plot(global_reach_max)

# Global scatter axis limits — fixed across tokens for smooth animation
pad = 0.5
global_x_min = flat_projections[:, :, 0].min() - pad
global_x_max = flat_projections[:, :, 0].max() + pad
global_y_min = flat_projections[:, :, 1].min() - pad
global_y_max = flat_projections[:, :, 1].max() + pad

cmap = plt.cm.get_cmap("tab10", n_prompts)
prompt_colors = [cmap(p / max(n_prompts - 1, 1)) for p in range(n_prompts)]

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 2.2, 1.5])
ax_scatter = fig.add_subplot(gs[0, 0])
ax_entropy = fig.add_subplot(gs[2, :])
table_cols = min(2, n_prompts)
table_rows = int(np.ceil(n_prompts / table_cols))
table_gs = gs[0, 1].subgridspec(table_rows, table_cols, hspace=0.6, wspace=0.3)
table_axes = []
for p in range(n_prompts):
    ax = fig.add_subplot(table_gs[p // table_cols, p % table_cols])
    ax.axis("off")
    table_axes.append(ax)
plt.subplots_adjust(bottom=0.12, top=0.9, hspace=0.45, wspace=0.3)

reach_cols = min(2, n_prompts)
reach_rows = int(np.ceil(n_prompts / reach_cols))
reach_gs = gs[1, :].subgridspec(reach_rows, reach_cols, hspace=0.35, wspace=0.25)
reach_axes = []
for p in range(n_prompts):
    ax = fig.add_subplot(reach_gs[p // reach_cols, p % reach_cols])
    reach_axes.append(ax)

# Left: UMAP colored by prompt labels
sc = ax_scatter.scatter(
    flat_projections[0, :, 0], flat_projections[0, :, 1],
    c=flat_prompt_ids, cmap=cmap, vmin=0, vmax=n_prompts - 1, s=12
)
cluster_count = int(metrics_cluster_counts[0].sum())
noise_count = int(metrics_noise_counts[0].sum())
ax_scatter.set_title(
    f"UMAP by prompt at token position 1 (OPTICS clusters={cluster_count}, noise={noise_count})",
    pad=26,
)
ax_scatter.set_xlabel("UMAP 1")
ax_scatter.set_ylabel("UMAP 2")
ax_scatter.set_xlim(global_x_min - pad, global_x_max + pad)
ax_scatter.set_ylim(global_y_min - pad, global_y_max + pad)
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=prompt_colors[i], markersize=8, label=label)
    for i, label in enumerate(unique_prompts)
]
ax_scatter.legend(
    handles=legend_handles,
    title="Prompt",
    fontsize=8,
    title_fontsize=9,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=max(1, min(3, n_prompts)),
)

# Metrics tables (top-right) - one per prompt.
def make_prompt_table_rows(t, p):
    pc = int(metrics_core_counts[t, p])
    pb = int(metrics_border_counts[t, p])
    pn = int(metrics_noise_counts[t, p])
    total = max(pc + pb + pn, 1)
    hom = float(metrics_core_homogeneity[t, p])
    cluster_count = int(metrics_cluster_counts[t, p])
    return [
        ["Clusters", str(cluster_count), "-"],
        ["Core", str(pc), f"{pc/total:.2f}"],
        ["Border", str(pb), f"{pb/total:.2f}"],
        ["Noise", str(pn), f"{pn/total:.2f}"],
        ["Homogeneity", "-", f"{hom:.2f}"],
    ]

table_holders = []
for p in range(n_prompts):
    table_axes[p].set_title(unique_prompts[p], fontsize=7, pad=3, color=prompt_colors[p], fontweight="bold")
    tbl = table_axes[p].table(
        cellText=make_prompt_table_rows(0, p),
        colLabels=["Metric", "N", "Share"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.2)
    table_holders.append(tbl)

# Prompt-specific reachability bars.
reach_bars_by_prompt = []
for p, ax in enumerate(reach_axes):
    vals = scaled_reachability[0, p]   # [num_return_per_prompt]
    x = np.arange(len(vals))
    bars = ax.bar(
        x,
        vals,
        width=1.0,
        align="center",
        color=prompt_colors[p],
        edgecolor="none",
    )
    reach_bars_by_prompt.append(bars)
    ax.set_title(f"Reachability - {unique_prompts[p]}", fontsize=9)
    ax.set_xlim(-0.5, max(len(vals) - 0.5, 0.5))
    ax.set_ylim(0, global_prompt_reach_max * 1.05)
    ax.set_xlabel("Ordered points")
    if p < reach_cols:
        ax.set_ylabel("Reachability (log1p)")

# Bottom: entropy per prompt over token position with current-token marker.
for prompt_text, prompt_curve in zip(entropy_prompt_texts, entropy_by_prompt):
    prompt_idx = unique_prompts.index(prompt_text) if prompt_text in unique_prompts else None
    line_color = cmap(prompt_idx) if prompt_idx is not None else "gray"
    ax_entropy.plot(entropy_tokens, prompt_curve, linewidth=1.5, color=line_color, label=prompt_text)

ax_entropy.set_xlim(1, entropy_by_prompt.shape[1])
ent_min = float(np.min(entropy_by_prompt))
ent_max = float(np.max(entropy_by_prompt))
if ent_max > ent_min:
    pad_y = (ent_max - ent_min) * 0.1
    ax_entropy.set_ylim(ent_min - pad_y, ent_max + pad_y)
ax_entropy.set_title("Entropy over token position by prompt")
ax_entropy.set_xlabel("Token position")
ax_entropy.set_ylabel("Entropy")
ax_entropy.legend(loc="upper right", fontsize=8)
current_token_line = ax_entropy.axvline(x=1, color="red", linestyle="--", linewidth=1.2)

# Slider for token position
ax_slider = plt.axes([0.2, 0.04, 0.6, 0.03])
slider = Slider(ax_slider, "Token", 1, num_tokens, valinit=1, valstep=1)


def update(_val):
    t = int(slider.val) - 1

    # Update scatter: flat_projections[t] is [num_sentences, 2]
    sc.set_offsets(flat_projections[t])
    ax_scatter.set_xlim(global_x_min, global_x_max)
    ax_scatter.set_ylim(global_y_min, global_y_max)
    cluster_count = int(metrics_cluster_counts[t].sum())
    noise_count = int(metrics_noise_counts[t].sum())
    ax_scatter.set_title(
        f"UMAP by prompt at token position {t + 1} (OPTICS clusters={cluster_count}, noise={noise_count})",
        pad=26,
    )

    # Recreate per-prompt metrics tables for token t.
    for p in range(n_prompts):
        table_axes[p].clear()
        table_axes[p].axis("off")
        table_axes[p].set_title(unique_prompts[p], fontsize=7, pad=3, color=prompt_colors[p], fontweight="bold")
        tbl = table_axes[p].table(
            cellText=make_prompt_table_rows(t, p),
            colLabels=["Metric", "N", "Share"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.2)
        table_holders[p] = tbl

    # Update reachability bars: scaled_reachability[t, p] is [num_return_per_prompt]
    for p, bars in enumerate(reach_bars_by_prompt):
        vals = scaled_reachability[t, p]
        for idx, bar in enumerate(bars):
            bar.set_height(vals[idx])
            bar.set_color(prompt_colors[p])
        reach_axes[p].set_xlim(-0.5, max(len(vals) - 0.5, 0.5))
        reach_axes[p].set_ylim(0, global_prompt_reach_max * 1.05)

    # Update entropy token indicator.
    token_pos = min(t + 1, entropy_by_prompt.shape[1])
    current_token_line.set_xdata([token_pos, token_pos])

    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
