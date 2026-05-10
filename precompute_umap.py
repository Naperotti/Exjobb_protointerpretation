import json
import numpy as np
import umap
from sklearn.cluster import OPTICS
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR, EMBEDDING_ARRAY_FILENAME, EMBEDDING_METADATA_FILENAME

# OPTICS clustering settings
MIN_SAMPLES = 10
MAX_EPS = 1
XI = 0.5
METRIC = "euclidean"

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# Load embeddings: [num_sentences, num_tokens, hidden_dim], float32
# Load metadata: list of dicts with prompt_text, sequence_index, model_name
embeddings = np.load(output_dir / EMBEDDING_ARRAY_FILENAME)
with open(output_dir / EMBEDDING_METADATA_FILENAME, "r", encoding="utf-8") as f:
    metadata = json.load(f)

num_sentences, num_tokens, hidden_dim = embeddings.shape
prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))
n_prompts = len(unique_prompts)

# prompt_masks[p]: bool array [num_sentences] — True for rows belonging to prompt p
prompt_masks = [np.array([t == unique_prompts[p] for t in prompt_texts]) for p in range(n_prompts)]
num_return_per_prompt = int(prompt_masks[0].sum())

print(f"Embeddings shape: {embeddings.shape}")
print(f"Prompts: {n_prompts}, returns per prompt: {num_return_per_prompt}")
print(f"Precomputing UMAP + OPTICS for {num_tokens} token positions...")


# input:  umap_2d [num_points, 2] — 2D UMAP projection for one prompt at one token position
# output: dict with per-point arrays (labels, ordering, reachability) and scalar counts
def fit_optics(umap_2d):
    model = OPTICS(min_samples=MIN_SAMPLES, max_eps=MAX_EPS, metric=METRIC, cluster_method="xi", xi=XI)
    labels = model.fit_predict(umap_2d)           # [num_points], int — cluster id or -1 for noise
    ordering = model.ordering_                    # [num_points], int — OPTICS traversal order
    reachability_raw = model.reachability_        # [num_points], float — reachability distances (may be inf)
    core_distances = model.core_distances_        # [num_points], float — inf for non-core points

    # Replace inf reachability with 1.0 so arrays are clean float32
    reachability_ordered = np.where(
        np.isfinite(reachability_raw[ordering]), reachability_raw[ordering], 1.0
    ).astype(np.float32)                          # [num_points], float32, in OPTICS order

    core_mask = np.isfinite(core_distances)       # [num_points], bool
    noise_mask = labels == -1                     # [num_points], bool
    border_mask = (~core_mask) & (~noise_mask)    # [num_points], bool

    core_count = int(core_mask.sum())
    border_count = int(border_mask.sum())
    noise_count = int(noise_mask.sum())
    n = max(len(labels), 1)
    cluster_count = len(set(labels[labels >= 0]))

    return {
        "labels": labels.astype(np.int32),        # [num_points]
        "ordering": ordering.astype(np.int64),    # [num_points]
        "reachability": reachability_ordered,      # [num_points], in OPTICS order
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "core_count": core_count,
        "border_count": border_count,
        "core_homogeneity": core_count / n,
    }


reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42)

# Output arrays — filled in the loop below
# projections:         [num_tokens, n_prompts, num_return_per_prompt, 2], float32
# optics_labels:       [num_tokens, n_prompts, num_return_per_prompt], int32
# optics_reachability: [num_tokens, n_prompts, num_return_per_prompt], float32
# optics_orderings:    [num_tokens, n_prompts, num_return_per_prompt], int64
# metrics_*:           [num_tokens, n_prompts], float32 or int32
projections         = np.zeros((num_tokens, n_prompts, num_return_per_prompt, 2), dtype=np.float32)
optics_labels       = np.zeros((num_tokens, n_prompts, num_return_per_prompt), dtype=np.int32)
optics_reachability = np.zeros((num_tokens, n_prompts, num_return_per_prompt), dtype=np.float32)
optics_orderings    = np.zeros((num_tokens, n_prompts, num_return_per_prompt), dtype=np.int64)
metrics_cluster_counts   = np.zeros((num_tokens, n_prompts), dtype=np.int32)
metrics_noise_counts     = np.zeros((num_tokens, n_prompts), dtype=np.int32)
metrics_core_counts      = np.zeros((num_tokens, n_prompts), dtype=np.int32)
metrics_border_counts    = np.zeros((num_tokens, n_prompts), dtype=np.int32)
metrics_core_homogeneity = np.zeros((num_tokens, n_prompts), dtype=np.float32)

for t in range(num_tokens):
    for p in range(n_prompts):
        # emb_p: [num_return_per_prompt, hidden_dim] — embeddings for prompt p at token t
        emb_p = embeddings[prompt_masks[p], t, :]

        # proj_p: [num_return_per_prompt, 2] — UMAP projection for this prompt
        proj_p = reducer.fit_transform(emb_p)
        projections[t, p] = proj_p.astype(np.float32)

        # optics_p: clustering result for this prompt's 2D projection
        optics_p = fit_optics(proj_p)
        optics_labels[t, p]       = optics_p["labels"]
        optics_orderings[t, p]    = optics_p["ordering"]
        optics_reachability[t, p] = optics_p["reachability"]
        metrics_cluster_counts[t, p]   = optics_p["cluster_count"]
        metrics_noise_counts[t, p]     = optics_p["noise_count"]
        metrics_core_counts[t, p]      = optics_p["core_count"]
        metrics_border_counts[t, p]    = optics_p["border_count"]
        metrics_core_homogeneity[t, p] = optics_p["core_homogeneity"]

    print(f"  Token {t + 1}/{num_tokens} done")

# Save all outputs
np.save(output_dir / "umap_projections.npy", projections)
np.save(output_dir / "optics_labels.npy", optics_labels)
np.save(output_dir / "optics_reachability.npy", optics_reachability)
np.save(output_dir / "optics_orderings.npy", optics_orderings)
np.savez_compressed(
    output_dir / "optics_metrics.npz",
    cluster_counts=metrics_cluster_counts,
    noise_counts=metrics_noise_counts,
    core_counts=metrics_core_counts,
    border_counts=metrics_border_counts,
    core_homogeneity=metrics_core_homogeneity,
)

print(f"Saved projections {projections.shape} to {output_dir}")
print(f"Saved OPTICS arrays {optics_labels.shape} to {output_dir}")
