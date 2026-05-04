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

fig, ax = plt.subplots(figsize=(10, 5))

for p in range(num_prompts):
    # Mean entropy across sequences for this prompt: [max_length]
    mean_ent = entropies[p].mean(axis=0)
    ax.plot(token_positions, mean_ent, label=f'"{prompt_texts[p]}"')

ax.set_xlabel("Token position")
ax.set_ylabel("Entropy (nats)")
ax.set_title("Mean per-token entropy during generation")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
