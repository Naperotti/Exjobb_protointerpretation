import numpy as np
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR, UMAP_PROJECTIONS_FILENAME, EMBEDDING_METADATA_FILENAME
import json

output_dir = Path(EMBEDDING_OUTPUT_DIR)

# projections: [num_tokens, n_prompts, num_return_per_prompt, 2]
projections = np.load(output_dir / UMAP_PROJECTIONS_FILENAME)
num_tokens, n_prompts, num_return_per_prompt, _ = projections.shape
print(f"Projections shape: {projections.shape}")
print(f"  num_tokens={num_tokens}, n_prompts={n_prompts}, num_return_per_prompt={num_return_per_prompt}")
print()

with open(output_dir / EMBEDDING_METADATA_FILENAME, "r", encoding="utf-8") as f:
    metadata = json.load(f)
prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))

# Check all token positions
print(f"{'Token':>6}  {'Prompt':>6}  {'Total':>8}  {'Unique 2D':>10}  {'Duplicates':>12}")
print("-" * 52)
for t in range(num_tokens):
    for p in range(n_prompts):
        pts = projections[t, p]  # [num_return_per_prompt, 2]
        unique = len(np.unique(pts.round(6), axis=0))
        duplicates = num_return_per_prompt - unique
        print(f"{t+1:>6}  {p:>6}  {num_return_per_prompt:>8}  {unique:>10}  {duplicates:>12}")
