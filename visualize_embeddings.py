import json
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from settings import EMBEDDING_OUTPUT_DIR
from pathlib import Path


def load_embeddings(output_dir):
    output_dir = Path(output_dir)
    embeddings = np.load(output_dir / "aligned_va_embeddings.npy")
    with open(output_dir / "aligned_va_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def sanity_check(embeddings):
    print(f"Shape: {embeddings.shape}")
    print(f"Any NaN: {np.isnan(embeddings).any()}")
    print(f"Any all-zero rows: {(np.abs(embeddings).sum(axis=1) == 0).any()}")
    print(f"Std across samples: {embeddings.std(axis=0).mean():.6f}")


def plot_umap(embeddings, labels, output_dir):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine")
    proj = reducer.fit_transform(embeddings)

    unique_labels = list(dict.fromkeys(labels))
    color_map = {label: i for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="tab10", s=60)

    handles = []
    for label in unique_labels:
        idx = color_map[label]
        color = plt.cm.tab10(idx / max(len(unique_labels) - 1, 1))
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label))
    plt.legend(handles=handles, title="Prompt", fontsize=8, title_fontsize=9)

    plt.title("UMAP of Aligned VA Embeddings")
    plt.tight_layout()
    output_path = Path(output_dir) / "umap_plot.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved UMAP plot to {output_path}")


def run_kmeans(embeddings, labels):
    n_clusters = len(set(labels))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, pred)
    print(f"KMeans clusters: {n_clusters}")
    print(f"Adjusted Rand Index: {ari:.3f}")


if __name__ == "__main__":
    embeddings, metadata = load_embeddings(EMBEDDING_OUTPUT_DIR)
    labels = [m["prompt_text"] for m in metadata]

    print("=== Sanity Checks ===")
    sanity_check(embeddings)

    print("\n=== UMAP ===")
    plot_umap(embeddings, labels, EMBEDDING_OUTPUT_DIR)

    print("\n=== KMeans Clustering ===")
    run_kmeans(embeddings, labels)