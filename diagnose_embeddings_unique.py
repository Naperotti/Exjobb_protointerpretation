import json
import numpy as np
from pathlib import Path
from settings import EMBEDDING_OUTPUT_DIR, EMBEDDING_ARRAY_FILENAME, EMBEDDING_METADATA_FILENAME

output_dir = Path(EMBEDDING_OUTPUT_DIR)

embeddings = np.load(output_dir / EMBEDDING_ARRAY_FILENAME)
with open(output_dir / EMBEDDING_METADATA_FILENAME, "r", encoding="utf-8") as f:
    metadata = json.load(f)

num_sentences, num_tokens, hidden_dim = embeddings.shape
prompt_texts = [m["prompt_text"] for m in metadata]
unique_prompts = list(dict.fromkeys(prompt_texts))
n_prompts = len(unique_prompts)
prompt_masks = [np.array([t == unique_prompts[p] for t in prompt_texts]) for p in range(n_prompts)]

print(f"Embeddings shape: {embeddings.shape}")
print(f"Prompts: {n_prompts}, sentences per prompt: {int(prompt_masks[0].sum())}")
print()

print(f"{'Token':>6}  {'Prompt':>6}  {'Total':>8}  {'Unique emb':>12}  {'Duplicates':>12}")
print("-" * 56)
for t in range(num_tokens):
    for p in range(n_prompts):
        emb_p = embeddings[prompt_masks[p], t, :]   # [num_return, hidden_dim]
        unique = len(np.unique(emb_p, axis=0))
        total = emb_p.shape[0]
        print(f"{t+1:>6}  {p:>6}  {total:>8}  {unique:>12}  {total - unique:>12}")
