import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import textwrap
from pathlib import Path

from settings import (
    EMBEDDING_OUTPUT_DIR,
    GENERATED_OUTPUT_PATH,
    EMBEDDING_METADATA_FILENAME,
    HOVER_TEXT_EXPORT_FILENAME,
    UMAP_PROJECTIONS_FILENAME,
    OPTICS_LABELS_FILENAME,
    OPTICS_REACHABILITY_FILENAME,
    OPTICS_ORDERINGS_FILENAME,
    OPTICS_METRICS_FILENAME,
)

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# --- Load all precomputed data from precompute_umap.py ---
projections = np.load(output_dir / UMAP_PROJECTIONS_FILENAME)
num_tokens, n_prompts, num_return_per_prompt, _ = projections.shape

optics_labels       = np.load(output_dir / OPTICS_LABELS_FILENAME)
optics_reachability = np.load(output_dir / OPTICS_REACHABILITY_FILENAME)
optics_orderings    = np.load(output_dir / OPTICS_ORDERINGS_FILENAME)

optics_metrics = np.load(output_dir / OPTICS_METRICS_FILENAME)
metrics_cluster_counts   = optics_metrics["cluster_counts"]

with open(output_dir / EMBEDDING_METADATA_FILENAME, "r", encoding="utf-8") as f:
    metadata = json.load(f)
prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))

entropy_data = np.load(GENERATED_OUTPUT_PATH, allow_pickle=False)
entropy_prompt_texts = [str(p) for p in entropy_data["prompt_texts"]]
entropy_by_prompt = entropy_data["entropies"].mean(axis=1)
entropy_tokens = np.arange(1, entropy_by_prompt.shape[1] + 1)

hover_data = np.load(output_dir / HOVER_TEXT_EXPORT_FILENAME, allow_pickle=False)
hover_full_texts = hover_data["full_texts"]
hover_char_end_offsets = hover_data["char_end_offsets"]
hover_metadata = json.loads(str(hover_data["metadata_json"]))

if int(hover_metadata.get("num_prompts", -1)) != n_prompts:
    raise ValueError(f"Hover export prompt count mismatch: {hover_metadata.get('num_prompts')} != {n_prompts}")
if int(hover_metadata.get("num_return", -1)) != num_return_per_prompt:
    raise ValueError(f"Hover export return count mismatch: {hover_metadata.get('num_return')} != {num_return_per_prompt}")
if hover_full_texts.shape != (n_prompts, num_return_per_prompt):
    raise ValueError(f"Hover text shape mismatch: {hover_full_texts.shape} != {(n_prompts, num_return_per_prompt)}")
if hover_char_end_offsets.shape != (n_prompts, num_return_per_prompt, entropy_by_prompt.shape[1]):
    raise ValueError(f"Hover offset shape mismatch: {hover_char_end_offsets.shape} != {(n_prompts, num_return_per_prompt, entropy_by_prompt.shape[1])}")

flat_sequence_ids = np.concatenate([np.arange(num_return_per_prompt) for _ in range(n_prompts)])

# --- Derive display arrays at load time ---
flat_projections = np.concatenate([projections[:, p, :, :] for p in range(n_prompts)], axis=1)
flat_prompt_ids = np.concatenate([np.full(num_return_per_prompt, p) for p in range(n_prompts)])
prompt_point_indices = [np.where(flat_prompt_ids == p)[0] for p in range(n_prompts)]

def scale_reachability_for_plot(vals):
    return np.log1p(vals)

scaled_reachability = scale_reachability_for_plot(optics_reachability)

global_reach_max = float(np.max(optics_reachability))
if not np.isfinite(global_reach_max) or global_reach_max <= 0:
    global_reach_max = 1.0
global_prompt_reach_max = scale_reachability_for_plot(global_reach_max)

pad = 0.5
global_x_min = flat_projections[:, :, 0].min() - pad
global_x_max = flat_projections[:, :, 0].max() + pad
global_y_min = flat_projections[:, :, 1].min() - pad
global_y_max = flat_projections[:, :, 1].max() + pad

tab10 = plt.get_cmap("tab10")
prompt_colors = [tab10.colors[p % len(tab10.colors)] for p in range(n_prompts)]
exact_cmap = ListedColormap(prompt_colors)

scatter_size = 2
current_token_index = 0

def format_hover_text(prompt_index, sequence_index, token_index):
    prefix_end = int(hover_char_end_offsets[prompt_index, sequence_index, token_index])
    prefix_text = hover_full_texts[prompt_index, sequence_index][:prefix_end]
    
    # Wrap long sequences so they expand downwards in lines, not horizontally
    wrapped_prefix = "\n".join(textwrap.wrap(prefix_text, width=60))
    
    return (
        f"Prompt: {unique_prompts[prompt_index]}\n"
        f"Sequence: {sequence_index + 1}\n"
        f"Token: {token_index + 1}\n\n"
        f"{wrapped_prefix}"
    )

fig = plt.figure(figsize=(24, 20), layout="constrained")
gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.5])

# Row 0-1: 3 Individual UMAP plots at the top
ax_umap_p0 = fig.add_subplot(gs[0:2, 0])
ax_umap_p1 = fig.add_subplot(gs[0:2, 1])
ax_umap_p2 = fig.add_subplot(gs[0:2, 2])

# Row 2-3: Shared UMAP takes up 2 height units on the right side
ax_umap_shared = fig.add_subplot(gs[2:4, 2])
umap_axes = [ax_umap_shared, ax_umap_p0, ax_umap_p1, ax_umap_p2]

scatter_artists = []

# Shared UMAP
sc_shared = ax_umap_shared.scatter(
    flat_projections[0, :, 0], flat_projections[0, :, 1],
    c=flat_prompt_ids, cmap=exact_cmap,
    vmin=-0.5, vmax=n_prompts - 0.5, s=scatter_size, alpha=0.8
)
scatter_artists.append(("shared", sc_shared, None))
ax_umap_shared.set_title("Shared UMAP (All Prompts)", fontsize=11, fontweight="bold")
ax_umap_shared.set_xlabel("UMAP 1", fontsize=9)
ax_umap_shared.set_ylabel("UMAP 2", fontsize=9)
ax_umap_shared.grid(True, alpha=0.3, linestyle="--")

# Individual UMAPs
for idx, (ax, prompt_idx) in enumerate(zip([ax_umap_p0, ax_umap_p1, ax_umap_p2], range(n_prompts))):
    prompt_mask = flat_prompt_ids == prompt_idx
    sc = ax.scatter(
        flat_projections[0, prompt_mask, 0], flat_projections[0, prompt_mask, 1],
        color=prompt_colors[prompt_idx], s=scatter_size, alpha=0.9
    )
    scatter_artists.append((f"prompt_{prompt_idx}", sc, prompt_idx))
    
    cluster_count = int(metrics_cluster_counts[0, prompt_idx])
    title_text = f"UMAP - {unique_prompts[prompt_idx]}"
    wrapped_title = "\n".join(textwrap.wrap(title_text, width=40))
    ax.set_title(f"{wrapped_title}\n[ Clusters: {cluster_count} ]", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

for ax in umap_axes:
    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)
    ax.set_aspect("equal", adjustable="box")

# Row 2-3: Reachability bars
reach_axes = []
reach_bars_by_prompt = []

reach_grid_positions = [
    gs[2, 0], # Reachability 1 
    gs[2, 1], # Reachability 2 
    gs[3, 0], # Reachability 3 
]

for p in range(n_prompts):
    pos = reach_grid_positions[p] if p < len(reach_grid_positions) else gs[3, 1]
    ax = fig.add_subplot(pos)
    reach_axes.append(ax)
    
    vals = scaled_reachability[0, p]
    x = np.arange(len(vals))
    bars = ax.bar(
        x, vals, width=1.0, align="center",
        color=prompt_colors[p], edgecolor="none", alpha=0.8
    )
    reach_bars_by_prompt.append(bars)
    
    title_text = f"Reachability - {unique_prompts[p]}"
    ax.set_title("\n".join(textwrap.wrap(title_text, width=45)), fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, max(len(vals) - 0.5, 0.5))
    ax.set_ylim(0, global_prompt_reach_max * 1.05)
    ax.set_xlabel("Ordered points", fontsize=8)
    ax.set_ylabel("Reachability (log1p)", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

# --- NEW: Static Text Box for Hover Information ---
ax_hover_text = fig.add_subplot(gs[3, 1])

# Formatera själva ritytan att se ut som en ruta
ax_hover_text.set_xticks([])
ax_hover_text.set_yticks([])
ax_hover_text.set_facecolor("#f8f9fa")
for spine in ax_hover_text.spines.values():
    spine.set_edgecolor("0.8")
    spine.set_linewidth(1.0)

DEFAULT_HOVER_TEXT = "Hover over a data point to view prompt details."

# Textobjektet existerar inuti den fasta ytan, utan en dynamisk bbox
hover_text_display = ax_hover_text.text(
    0.05, 0.95, DEFAULT_HOVER_TEXT,
    transform=ax_hover_text.transAxes,
    fontsize=8, va="top", ha="left"
)
# --------------------------------------------------

# Row 4: Entropy 
ax_entropy = fig.add_subplot(gs[4, :])
for prompt_text, prompt_curve in zip(entropy_prompt_texts, entropy_by_prompt):
    prompt_idx = unique_prompts.index(prompt_text) if prompt_text in unique_prompts else None
    line_color = prompt_colors[prompt_idx] if prompt_idx is not None else "gray"
    ax_entropy.plot(entropy_tokens, prompt_curve, linewidth=2, color=line_color, label=prompt_text, alpha=0.8)

ax_entropy.set_xlim(1, entropy_by_prompt.shape[1])
ent_min = float(np.min(entropy_by_prompt))
ent_max = float(np.max(entropy_by_prompt))
if ent_max > ent_min:
    pad_y = (ent_max - ent_min) * 0.1
    ax_entropy.set_ylim(ent_min - pad_y, ent_max + pad_y)
ax_entropy.set_title("Entropy over token position by prompt", fontsize=11, fontweight="bold")
ax_entropy.set_xlabel("Token position", fontsize=9)
ax_entropy.set_ylabel("Entropy", fontsize=9)
ax_entropy.grid(True, alpha=0.3, linestyle="--")
ax_entropy.legend(loc="upper right", fontsize=9)
current_token_line = ax_entropy.axvline(x=1, color="red", linestyle="--", linewidth=1.5)

ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
slider = Slider(ax_slider, "Token", 1, num_tokens, valinit=1, valstep=1)

def on_move(event):
    current_ax = event.inaxes
    if current_ax not in umap_axes:
        if hover_text_display.get_text() != DEFAULT_HOVER_TEXT:
            hover_text_display.set_text(DEFAULT_HOVER_TEXT)
            fig.canvas.draw_idle()
        return
    
    ax_to_scatter = {
        ax_umap_shared: scatter_artists[0][1],
        ax_umap_p0: scatter_artists[1][1],
        ax_umap_p1: scatter_artists[2][1],
        ax_umap_p2: scatter_artists[3][1],
    }
    
    sc = ax_to_scatter[current_ax]
    contains, info = sc.contains(event)
    inds = info.get("ind", [])
    
    if (not contains) or len(inds) == 0:
        if hover_text_display.get_text() != DEFAULT_HOVER_TEXT:
            hover_text_display.set_text(DEFAULT_HOVER_TEXT)
            fig.canvas.draw_idle()
        return
    
    flat_index = int(inds[0])
    if current_ax == ax_umap_shared:
        prompt_index = int(flat_prompt_ids[flat_index])
        sequence_index = int(flat_sequence_ids[flat_index])
    else:
        prompt_index = next(idx for idx, axis in enumerate([ax_umap_p0, ax_umap_p1, ax_umap_p2]) if axis == current_ax)
        sequence_index = int(flat_sequence_ids[prompt_point_indices[prompt_index][flat_index]])
    
    new_text = format_hover_text(prompt_index, sequence_index, current_token_index)
    if hover_text_display.get_text() != new_text:
        hover_text_display.set_text(new_text)
        fig.canvas.draw_idle()

def update(_val):
    global current_token_index
    t = int(slider.val) - 1
    current_token_index = t
    
    for label, sc, prompt_idx in scatter_artists:
        if label == "shared":
            sc.set_offsets(flat_projections[t])
        else:
            sc.set_offsets(flat_projections[t, prompt_point_indices[prompt_idx]])
            
            ax = sc.axes
            cluster_count = int(metrics_cluster_counts[t, prompt_idx])
            title_text = f"UMAP - {unique_prompts[prompt_idx]}"
            wrapped_title = "\n".join(textwrap.wrap(title_text, width=40))
            ax.set_title(f"{wrapped_title}\n[ Clusters: {cluster_count} ]", fontsize=10, fontweight="bold")
    
    for p, bars in enumerate(reach_bars_by_prompt):
        vals = scaled_reachability[t, p]
        for idx, bar in enumerate(bars):
            bar.set_height(vals[idx])
    
    token_pos = min(t + 1, entropy_by_prompt.shape[1])
    current_token_line.set_xdata([token_pos, token_pos])
    
    # Reset text box when the timeline changes
    hover_text_display.set_text(DEFAULT_HOVER_TEXT)
    
    fig.canvas.draw_idle()

slider.on_changed(update)
fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()