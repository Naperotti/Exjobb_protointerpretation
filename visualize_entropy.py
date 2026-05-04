import numpy as np
import matplotlib.pyplot as plt
from settings import GENERATED_OUTPUT_PATH

# Load saved data
# entropies: [num_prompts, num_return, max_length], float32
# prompt_texts: [num_prompts], str
data = np.load(GENERATED_OUTPUT_PATH, allow_pickle=False)
entropies = data["entropies"]
prompt_texts = data["prompt_texts"]

num_prompts, num_return, max_length = entropies.shape
token_positions = range(1, max_length + 1)

fig, axes = plt.subplots(num_prompts, 1, figsize=(10, 3 * num_prompts), sharex=True)
if num_prompts == 1:
    axes = [axes]

for p, ax in enumerate(axes):
    for s in range(num_return):
        ax.plot(token_positions, entropies[p, s], alpha=0.6, label=f"seq {s + 1}")

    # Mean entropy across sequences for this prompt
    mean_ent = entropies[p].mean(axis=0)  # [max_length]
    ax.plot(token_positions, mean_ent, color="black", linewidth=2, label="mean")

    ax.set_title(f'"{prompt_texts[p]}"', fontsize=9)
    ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Token position")
plt.suptitle("Per-token entropy during generation", fontsize=12)
plt.tight_layout()
plt.show()
