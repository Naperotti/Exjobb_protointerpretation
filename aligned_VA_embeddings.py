import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from settings import EMBEDDING_INPUT_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_OUTPUT_DIR

# Which transformer layers to extract hidden states from and average together.
# Chosen empirically for each model size — later layers carry more semantic content.
# För 8B modellen
# SELECTED_LAYERS = [26, 27, 29, 30, 31]

# För 1.7B modellen
SELECTED_LAYERS = [19, 20, 21, 22, 23]


# Loads generated token sequences from the .npz produced by generate.py.
# Returns a dict with "settings" (generation hyperparams) and "prompts"
# (list of dicts with "prompt" text and "sequences" dict of token ID arrays).
def load_sequences(input_path):
	input_path = Path(input_path)
	data = np.load(input_path, allow_pickle=False)

	settings = json.loads(str(data["settings_json"]))
	prompt_texts = data["prompt_texts"]
	# sequences: [num_prompts, num_return, max_length], int64
	sequences = data["sequences"]

	prompts = []
	for i, prompt_text in enumerate(prompt_texts):
		seq_dict = {}
		for j in range(sequences.shape[1]):
			seq_dict[str(j + 1)] = sequences[i, j]  # [max_length], int64 token IDs
		prompts.append({"prompt": str(prompt_text), "sequences": seq_dict})

	return {"settings": settings, "prompts": prompts}


# PromptEOL-style template: wraps the generated sentence so the model sees it
# as a completion task. {sentence} is replaced with the token IDs of the sentence
# prefix at each extraction step.
FUTURE_EOL_TEMPLATE = "Forecasting the subsequent tokens {sentence} in one word:" #Visade artikeln att PromptEOL är bättre?


# Loads the model and registers forward hooks on the self-attention output of
# each selected layer. Hooks capture the last-token hidden state on every forward pass.
# Returns tokenizer, model, layer_outputs dict (filled by hooks), and hook handles.
def setup_model_and_hooks(model_name, selected_layers):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	model.eval()

	layers = model.model.layers
	# layer_outputs[layer_index]: float32 tensor, shape [hidden_dim], filled after each forward pass
	layer_outputs = {layer_index: None for layer_index in selected_layers}
	hooks = []

	def make_hook(layer_index):
		def hook(_module, _inputs, output):
			# output[0]: [batch, seq_len, hidden_dim] — take only the last token position
			attention_output = output[0] if isinstance(output, tuple) else output
			layer_outputs[layer_index] = attention_output[:, -1, :].detach()  # [1, hidden_dim]
		return hook

	for layer_index in selected_layers:
		hooks.append(layers[layer_index].self_attn.register_forward_hook(make_hook(layer_index)))

	return tokenizer, model, layer_outputs, hooks


# Extracts a single embedding vector for a sentence prefix of length n.
# Builds input: [prefix_ids] + [sentence_ids_prefix (first n tokens)] + [suffix_ids]
# Runs one forward pass, reads the last-token hidden state from each selected layer,
# then mean-pools across layers.
# Returns: float32 numpy array, shape [hidden_dim]
def extract_token_embedding(model, layer_outputs, selected_layers, prefix_ids, sentence_ids_prefix, suffix_ids):
	current_ids = prefix_ids + sentence_ids_prefix + suffix_ids
	# input_ids: [1, total_len], int64
	input_ids = torch.tensor([current_ids], dtype=torch.long)

	for layer_index in selected_layers:
		layer_outputs[layer_index] = None

	with torch.no_grad():
		_ = model(input_ids=input_ids, use_cache=False)

	# Stack selected layer outputs: [num_layers, hidden_dim]
	stacked = torch.stack(
		[layer_outputs[layer_index].squeeze(0) for layer_index in selected_layers],
		dim=0,
	)
	# Mean across layers: [hidden_dim], float32
	return stacked.mean(dim=0).float().cpu().numpy()


# Saves embeddings and per-sentence metadata to disk.
# embeddings: [num_sentences, num_tokens, hidden_dim], float32 -> aligned_va_embeddings.npy
# metadata: list of dicts with prompt_text, sequence_index, model_name -> aligned_va_metadata.json
def save_embeddings(embeddings, metadata, output_dir):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	np.save(output_dir / "aligned_va_embeddings.npy", embeddings)
	with open(output_dir / "aligned_va_metadata.json", "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=False)
	print(f"Saved embeddings {embeddings.shape} to {output_dir}")


# Reloads saved files and prints shapes as a quick sanity check.
def print_saved_embedding_shapes(output_dir):
	output_dir = Path(output_dir)
	loaded_embeddings = np.load(output_dir / "aligned_va_embeddings.npy")
	with open(output_dir / "aligned_va_metadata.json", "r", encoding="utf-8") as f:
		loaded_metadata = json.load(f)

	print("Embedding sanity check:")
	print(f"- aligned_va_embeddings shape: {loaded_embeddings.shape}")
	print(f"- metadata entries: {len(loaded_metadata)}")


if __name__ == "__main__":
	data = load_sequences(EMBEDDING_INPUT_PATH)

	model_name = EMBEDDING_MODEL_NAME or data["settings"]["model"]
	prompt_groups = data["prompts"]

	# Split template into prefix/suffix around {sentence} placeholder, tokenize once
	prefix_text, suffix_text = FUTURE_EOL_TEMPLATE.split("{sentence}")
	tokenizer, model, layer_outputs, hooks = setup_model_and_hooks(model_name, SELECTED_LAYERS)
	# prefix_ids / suffix_ids: list of int token IDs (no special tokens added)
	prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
	suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]

	# Flatten all sequences across prompts into one list for embedding extraction
	all_sentence_ids = []
	metadata = []
	for prompt_group in prompt_groups:
		prompt_text = prompt_group["prompt"]
		for sequence_index, sentence_token_ids in prompt_group["sequences"].items():
			all_sentence_ids.append(sentence_token_ids)
			metadata.append(
				{
					"prompt_text": prompt_text,
					"sequence_index": int(sequence_index),
					"model_name": model_name,
				}
			)

	num_sentences = len(all_sentence_ids)
	num_tokens = len(all_sentence_ids[0])
	hidden_dim = model.config.hidden_size
	# embeddings[s, t]: embedding of sentence s after seeing t+1 generated tokens
	# shape: [num_sentences, num_tokens, hidden_dim], float32
	embeddings = np.zeros([num_sentences, num_tokens, hidden_dim], dtype=np.float32)

	for sentence_idx, sentence_ids in enumerate(all_sentence_ids):
		for n in range(1, num_tokens + 1):
			# Pass the first n tokens of the sentence as the sentence prefix
			embeddings[sentence_idx, n - 1] = extract_token_embedding(
				model, layer_outputs, SELECTED_LAYERS,
				prefix_ids, sentence_ids[:n].tolist(), suffix_ids,
			)
		print(f"Extracted embeddings for sentence {sentence_idx + 1}/{num_sentences}")

	# Remove hooks to avoid memory leaks
	for hook in hooks:
		hook.remove()

	save_embeddings(embeddings, metadata, EMBEDDING_OUTPUT_DIR)
	print_saved_embedding_shapes(EMBEDDING_OUTPUT_DIR)