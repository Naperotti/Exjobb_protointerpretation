import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import OPTICS
from matplotlib.lines import Line2D
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR, GENERATED_OUTPUT_PATH

# Load precomputed UMAP projections from precompute_umap.py
# projections: [num_tokens, num_sentences, 2]
output_dir = Path(EMBEDDING_OUTPUT_DIR)
projections = np.load(output_dir / "umap_projections.npy")
num_tokens = projections.shape[0]
with open(output_dir / "aligned_va_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))
prompt_ids = np.array([unique_prompts.index(p) for p in prompt_texts])
n_prompts = len(unique_prompts)

# Load entropy curve for bottom panel.
entropy_data = np.load(GENERATED_OUTPUT_PATH, allow_pickle=False)
# entropies: [num_prompts, num_return, max_length]
entropy_prompt_texts = [str(p) for p in entropy_data["prompt_texts"]]
entropy_by_prompt = entropy_data["entropies"].mean(axis=1)
entropy_tokens = np.arange(1, entropy_by_prompt.shape[1] + 1)

# OPTICS settings
MIN_SAMPLES = 10
MAX_EPS = 1
XI = 0.5
METRIC = "euclidean"


def fit_optics(umap_data):
    model = OPTICS(
        min_samples=MIN_SAMPLES,
        max_eps=MAX_EPS,
        metric=METRIC,
        cluster_method="xi",
        xi=XI,
    )
    labels = model.fit_predict(umap_data)
    ordering = model.ordering_
    reachability_raw = model.reachability_
    core_distances = model.core_distances_
    reachability_ordered = reachability_raw[ordering]

    core_mask = np.isfinite(core_distances)
    border_mask = (~core_mask) & np.isfinite(reachability_raw)
    noise_mask = (~core_mask) & (~np.isfinite(reachability_raw))

    core_count = int(core_mask.sum())
    border_count = int(border_mask.sum())
    noise_count = int(noise_mask.sum())
    sample_count = int(len(labels))
    core_homogeneity = core_count / sample_count if sample_count > 0 else 0.0

    return {
        "labels": labels,
        "ordering": ordering,
        "reachability_ordered": reachability_ordered,
        "core_mask": core_mask,
        "noise_mask": noise_mask,
        "core_count": core_count,
        "border_count": border_count,
        "noise_count": noise_count,
        "sample_count": sample_count,
        "core_homogeneity": core_homogeneity,
    }


print(f"Precomputing OPTICS for {num_tokens} token positions...")
all_labels = []
all_reachability = []
all_orderings = []
all_ordered_prompt_ids = []
all_ordered_colors = []
cmap = plt.cm.get_cmap("tab10", n_prompts)
prompt_colors = [cmap(p / max(n_prompts - 1, 1)) for p in range(n_prompts)]
all_core_counts = []
all_border_counts = []
all_noise_counts = []
all_core_homogeneity = []
all_prompt_reachability = []
all_cluster_counts = []
all_noise_label_counts = []
all_prompt_core_counts = []
all_prompt_border_counts = []
all_prompt_noise_counts = []
all_prompt_homogeneity = []
all_prompt_cluster_counts = []

for t in range(num_tokens):
    optics = fit_optics(projections[t])

    labels = optics["labels"]
    ordering = optics["ordering"]
    reachability = optics["reachability_ordered"]
    reachability_finite_one = np.where(np.isfinite(reachability), reachability, 1.0)

    all_labels.append(labels)
    all_reachability.append(reachability_finite_one)
    all_orderings.append(ordering)

    ordered_prompt_ids = prompt_ids[ordering]
    all_ordered_prompt_ids.append(ordered_prompt_ids)
    all_ordered_colors.append(cmap(ordered_prompt_ids.astype(float)))

    all_core_counts.append(optics["core_count"])
    all_border_counts.append(optics["border_count"])
    all_noise_counts.append(optics["noise_count"])
    all_core_homogeneity.append(optics["core_homogeneity"])

    core_mask_t = optics["core_mask"]
    noise_mask_t = optics["noise_mask"]
    border_mask_t = (~core_mask_t) & (~noise_mask_t)

    # Assign each non-noise cluster to exactly one prompt (majority membership)
    # so clusters are not double counted across prompt tables.
    assigned_cluster_counts = [0] * n_prompts
    for cluster_id in np.unique(labels[labels >= 0]):
        cluster_mask = labels == cluster_id
        votes = np.bincount(prompt_ids[cluster_mask], minlength=n_prompts)
        owner_prompt = int(np.argmax(votes))
        assigned_cluster_counts[owner_prompt] += 1

    prompt_core_counts = []
    prompt_border_counts = []
    prompt_noise_counts = []
    prompt_homogeneity = []
    prompt_cluster_counts = []
    for p in range(n_prompts):
        pm = (prompt_ids == p)
        pc = int((core_mask_t & pm).sum())
        pb = int((border_mask_t & pm).sum())
        pn = int((noise_mask_t & pm).sum())
        pt = max(pc + pb + pn, 1)
        prompt_core_counts.append(pc)
        prompt_border_counts.append(pb)
        prompt_noise_counts.append(pn)
        prompt_homogeneity.append(pc / pt)
        prompt_cluster_counts.append(assigned_cluster_counts[p])
    all_prompt_core_counts.append(prompt_core_counts)
    all_prompt_border_counts.append(prompt_border_counts)
    all_prompt_noise_counts.append(prompt_noise_counts)
    all_prompt_homogeneity.append(prompt_homogeneity)
    all_prompt_cluster_counts.append(prompt_cluster_counts)

    prompt_reachability = []
    for p in range(n_prompts):
        prompt_vals = reachability_finite_one[ordered_prompt_ids == p]
        prompt_reachability.append(prompt_vals)
    all_prompt_reachability.append(prompt_reachability)

    all_cluster_counts.append(len(set(labels[labels >= 0])))
    all_noise_label_counts.append(int(np.sum(labels == -1)))
    print(f"  Token {t + 1}/{num_tokens} done")


def scale_reachability_for_plot(vals):
    return np.log1p(vals)

# Save derived dataset: reachability with inf values replaced by 1.
reachability_dataset = np.array(all_reachability, dtype=np.float32)
reachability_output_path = output_dir / "optics_reachability_inf_as_one.npy"
np.save(reachability_output_path, reachability_dataset)
print(f"Saved reachability dataset to {reachability_output_path}")

# Initial state (token position 1)
initial_proj = projections[0]
initial_labels = all_labels[0]
initial_reachability = all_reachability[0]
initial_ordered_prompt_ids = all_ordered_prompt_ids[0]

# Global fixed axis limits for smooth token-to-token comparison.
global_x_min = projections[:, :, 0].min()
global_x_max = projections[:, :, 0].max()
global_y_min = projections[:, :, 1].min()
global_y_max = projections[:, :, 1].max()
pad = 0.5

# Keep reachability scale fixed across tokens.
global_reach_max = np.max(reachability_dataset)
if not np.isfinite(global_reach_max) or global_reach_max <= 0:
    global_reach_max = 1.0

scaled_prompt_reachability = []
for token_vals in all_prompt_reachability:
    scaled_prompt_reachability.append([scale_reachability_for_plot(v) for v in token_vals])

global_prompt_reach_max = scale_reachability_for_plot(MAX_EPS + 1)

prompt_counts = [(prompt_ids == p).sum() for p in range(n_prompts)]

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
    initial_proj[:, 0], initial_proj[:, 1],
    c=prompt_ids, cmap=cmap, vmin=0, vmax=n_prompts - 1, s=12
)
cluster_count = all_cluster_counts[0]
noise_count = all_noise_label_counts[0]
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
    pc = all_prompt_core_counts[t][p]
    pb = all_prompt_border_counts[t][p]
    pn = all_prompt_noise_counts[t][p]
    total = max(pc + pb + pn, 1)
    hom = all_prompt_homogeneity[t][p]
    cluster_count = all_prompt_cluster_counts[t][p]
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
    vals = scaled_prompt_reachability[0][p]
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
    proj = projections[t]
    labels = all_labels[t]
    ordered_colors = all_ordered_colors[t]

    # Update scatter positions
    sc.set_offsets(proj)

    # Keep scatter limits fixed so movement reflects data, not axis rescaling.
    ax_scatter.set_xlim(global_x_min - pad, global_x_max + pad)
    ax_scatter.set_ylim(global_y_min - pad, global_y_max + pad)
    cluster_count = all_cluster_counts[t]
    noise_count = all_noise_label_counts[t]
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

    # Update one reachability plot per prompt.
    for p, bars in enumerate(reach_bars_by_prompt):
        vals = scaled_prompt_reachability[t][p]
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
