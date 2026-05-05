import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# Load precomputed UMAP projections and ARI scores from precompute_umap.py
# projections: [num_tokens, num_sentences, 2]
# ari_scores:  [num_tokens]
projections = np.load(output_dir / "umap_projections.npy")
ari_scores = np.load(output_dir / "umap_ari_scores.npy")
with open(output_dir / "aligned_va_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

labels = [m["prompt_text"] for m in metadata]
unique_labels = list(dict.fromkeys(labels))
label_ids = np.array([unique_labels.index(l) for l in labels])
n_clusters = len(unique_labels)
num_tokens = projections.shape[0]

cmap = plt.cm.get_cmap("tab10", n_clusters)

fig, (ax_scatter, ax_ari) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.18)

sc = ax_scatter.scatter(
    projections[0, :, 0], projections[0, :, 1],
    c=label_ids, cmap="tab10", vmin=0, vmax=n_clusters - 1, s=60
)
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=8, label=lbl)
    for i, lbl in enumerate(unique_labels)
]
ax_scatter.legend(handles=handles, title="Prompt", fontsize=8, title_fontsize=9)
ax_scatter.set_title(f"UMAP at token position 1  (ARI={ari_scores[0]:.3f})")
ax_scatter.set_xlabel("UMAP 1")
ax_scatter.set_ylabel("UMAP 2")

ax_ari.plot(range(1, num_tokens + 1), ari_scores, color="steelblue")
ax_ari.set_xlabel("Token position")
ax_ari.set_ylabel("Adjusted Rand Index")
ax_ari.set_title("KMeans ARI over token position")
ax_ari.set_xlim(1, num_tokens)
vline = ax_ari.axvline(x=1, color="red", linestyle="--")

ax_slider = plt.axes([0.15, 0.06, 0.7, 0.03])
slider = Slider(ax_slider, "Token", 1, num_tokens, valinit=1, valstep=1)


def update(val):
    t = int(slider.val) - 1
    proj = projections[t]
    sc.set_offsets(proj)
    pad = 0.5
    ax_scatter.set_xlim(proj[:, 0].min() - pad, proj[:, 0].max() + pad)
    ax_scatter.set_ylim(proj[:, 1].min() - pad, proj[:, 1].max() + pad)
    ax_scatter.set_title(f"UMAP at token position {t + 1}  (ARI={ari_scores[t]:.3f})")
    vline.set_xdata([t + 1])
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
