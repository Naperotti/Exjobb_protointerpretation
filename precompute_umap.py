import json
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# Load embeddings and metadata
# embeddings: [num_sentences, num_tokens, hidden_dim], float32
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

# projections: [num_tokens, num_sentences, 2]
# ari_scores:  [num_tokens]
np.save(output_dir / "umap_projections.npy", np.array(projections, dtype=np.float32))
np.save(output_dir / "umap_ari_scores.npy", np.array(ari_scores, dtype=np.float32))
print(f"Saved projections {np.array(projections).shape} and ARI scores to {output_dir}")
