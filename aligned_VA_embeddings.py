import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from settings import EMBEDDING_INPUT_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_OUTPUT_DIR

# För 8B modellen
# SELECTED_LAYERS = [26, 27, 29, 30, 31]

# För 1.7B modellen
SELECTED_LAYERS = [19, 20, 21, 22, 23]


def load_sequences(input_path):
	input_path = Path(input_path)
	data = np.load(input_path, allow_pickle=False)

	settings = json.loads(str(data["settings_json"]))
	prompt_texts = data["prompt_texts"]
	sequences = data["sequences"]

	prompts = []
	for i, prompt_text in enumerate(prompt_texts):
		seq_dict = {}
		for j in range(sequences.shape[1]):
			seq_dict[str(j + 1)] = sequences[i, j]  # numpy int64 array of token IDs
		prompts.append({"prompt": str(prompt_text), "sequences": seq_dict})

	return {"settings": settings, "prompts": prompts}


FUTURE_EOL_TEMPLATE = "Forecasting the subsequent tokens {sentence} in one word:" #Visade artikeln att PromptEOL är bättre?



def setup_model_and_hooks(model_name, selected_layers):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	model.eval()

	layers = model.model.layers
	layer_outputs = {layer_index: None for layer_index in selected_layers}
	hooks = []

	def make_hook(layer_index):
		def hook(_module, _inputs, output):
			attention_output = output[0] if isinstance(output, tuple) else output
			layer_outputs[layer_index] = attention_output[:, -1, :].detach()
		return hook

	for layer_index in selected_layers:
		hooks.append(layers[layer_index].self_attn.register_forward_hook(make_hook(layer_index)))

	return tokenizer, model, layer_outputs, hooks


def extract_token_embedding(model, layer_outputs, selected_layers, prefix_ids, sentence_ids_prefix, suffix_ids):
	current_ids = prefix_ids + sentence_ids_prefix + suffix_ids
	input_ids = torch.tensor([current_ids], dtype=torch.long)

	for layer_index in selected_layers:
		layer_outputs[layer_index] = None

	with torch.no_grad():
		_ = model(input_ids=input_ids, use_cache=False)

	stacked = torch.stack(
		[layer_outputs[layer_index].squeeze(0) for layer_index in selected_layers],
		dim=0,
	)
	return stacked.mean(dim=0).float().cpu().numpy()


def save_embeddings(embeddings, metadata, output_dir):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	np.save(output_dir / "aligned_va_embeddings.npy", embeddings)
	with open(output_dir / "aligned_va_metadata.json", "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=False)
	print(f"Saved embeddings {embeddings.shape} to {output_dir}")


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

	# Tokenize template prefix and suffix once
	prefix_text, suffix_text = FUTURE_EOL_TEMPLATE.split("{sentence}")
	tokenizer, model, layer_outputs, hooks = setup_model_and_hooks(model_name, SELECTED_LAYERS)
	prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
	suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]

	# Collect all sentences and metadata
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
	embeddings = np.zeros([num_sentences, num_tokens, hidden_dim], dtype=np.float32)

	for sentence_idx, sentence_ids in enumerate(all_sentence_ids):
		for n in range(1, num_tokens + 1):
			embeddings[sentence_idx, n - 1] = extract_token_embedding(
				model, layer_outputs, SELECTED_LAYERS,
				prefix_ids, sentence_ids[:n].tolist(), suffix_ids,
			)
		print(f"Extracted embeddings for sentence {sentence_idx + 1}/{num_sentences}")

	for hook in hooks:
		hook.remove()

	save_embeddings(embeddings, metadata, EMBEDDING_OUTPUT_DIR)
	print_saved_embedding_shapes(EMBEDDING_OUTPUT_DIR)