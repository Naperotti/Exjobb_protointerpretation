import json
import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR

# Load embeddings and metadata
# embeddings: [num_sentences, num_tokens, hidden_dim], float32
output_dir = Path(EMBEDDING_OUTPUT_DIR)
embeddings = np.load(output_dir / "aligned_va_embeddings.npy")
with open(output_dir / "aligned_va_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

num_sentences, num_tokens, hidden_dim = embeddings.shape
labels = [m["prompt_text"] for m in metadata]
unique_labels = list(dict.fromkeys(labels))
label_ids = np.array([unique_labels.index(l) for l in labels])
n_clusters = len(unique_labels)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Precomputing UMAP for {num_tokens} token positions...")

# Precompute UMAP projection and KMeans ARI for every token position upfront.
# projections: [num_tokens, num_sentences, 2]
# ari_scores:  [num_tokens]
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42)
projections = []
ari_scores = []
for t in range(num_tokens):
    proj = reducer.fit_transform(embeddings[:, t, :])  # [num_sentences, 2]
    projections.append(proj)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(embeddings[:, t, :])
    ari_scores.append(adjusted_rand_score(label_ids, pred))
    print(f"  Token {t + 1}/{num_tokens} done (ARI={ari_scores[-1]:.3f})")

projections = np.array(projections)  # [num_tokens, num_sentences, 2]
ari_scores = np.array(ari_scores)    # [num_tokens]

cmap = plt.cm.get_cmap("tab10", n_clusters)

# Figure: UMAP scatter on the left, ARI over token position on the right
fig, (ax_scatter, ax_ari) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.18)

# Initial scatter at token position 0
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

# ARI line with a red dashed marker tracking current token
ax_ari.plot(range(1, num_tokens + 1), ari_scores, color="steelblue")
ax_ari.set_xlabel("Token position")
ax_ari.set_ylabel("Adjusted Rand Index")
ax_ari.set_title("KMeans ARI over token position")
ax_ari.set_xlim(1, num_tokens)
vline = ax_ari.axvline(x=1, color="red", linestyle="--")

# Slider at the bottom to scrub through token positions
ax_slider = plt.axes([0.15, 0.06, 0.7, 0.03])
slider = Slider(ax_slider, "Token", 1, num_tokens, valinit=1, valstep=1)


def update(val):
    t = int(slider.val) - 1  # convert to 0-indexed
    proj = projections[t]
    sc.set_offsets(proj)
    # Rescale axes to the new projection extent
    pad = 0.5
    ax_scatter.set_xlim(proj[:, 0].min() - pad, proj[:, 0].max() + pad)
    ax_scatter.set_ylim(proj[:, 1].min() - pad, proj[:, 1].max() + pad)
    ax_scatter.set_title(f"UMAP at token position {t + 1}  (ARI={ari_scores[t]:.3f})")
    vline.set_xdata([t + 1])
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
