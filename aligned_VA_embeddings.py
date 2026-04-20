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
	with open(input_path, "r", encoding="utf-8") as f:
		return json.load(f)


FUTURE_EOL_TEMPLATE = "Forecasting the subsequent tokens {sentence} in one word:"



def extract_aligned_va_embeddings(texts, model_name, selected_layers):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	model.eval()

	layers = model.model.layers

	layer_last_token_outputs = {layer_index: None for layer_index in selected_layers}
	hooks = []

	def make_hook(layer_index):
		def hook(_module, _inputs, output):
			attention_output = output[0] if isinstance(output, tuple) else output
			layer_last_token_outputs[layer_index] = attention_output[:, -1, :].detach()

		return hook

	for layer_index in selected_layers:
		hooks.append(layers[layer_index].self_attn.register_forward_hook(make_hook(layer_index)))

	embeddings = []
	for i, text in enumerate(texts):
		prompted_text = FUTURE_EOL_TEMPLATE.format(sentence=text) # use FUTURE_EOL_TEMPLATE to create a prompt that encourages the model to focus on the last token
		inputs = tokenizer(prompted_text, return_tensors="pt", truncation=True, max_length=512)

		for layer_index in selected_layers:
			layer_last_token_outputs[layer_index] = None

		with torch.no_grad():
			_ = model(**inputs, use_cache=False)

		stacked_layer_outputs = torch.stack(
			[layer_last_token_outputs[layer_index].squeeze(0) for layer_index in selected_layers],
			dim=0,
		)
		sentence_embedding = stacked_layer_outputs.mean(dim=0).float().cpu().numpy()
		embeddings.append(sentence_embedding)
		print(f"Extracted aligned VA embedding for sequence {i + 1}/{len(texts)}")

	for hook in hooks:
		hook.remove()

	return np.array(embeddings)


def save_embeddings(embeddings, metadata, output_dir):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	np.save(output_dir / "aligned_va_embeddings.npy", embeddings)
	with open(output_dir / "aligned_va_metadata.json", "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=False)
	print(f"Saved embeddings {embeddings.shape} to {output_dir}")


if __name__ == "__main__":
	data = load_sequences(EMBEDDING_INPUT_PATH)

	model_name = EMBEDDING_MODEL_NAME or data["settings"]["model"]
	prompt_groups = data["prompts"]

	texts = []
	metadata = []

	for prompt_group in prompt_groups:
		prompt_text = prompt_group["prompt"]
		for sequence_index, generated_text in prompt_group["sequences"].items():
			texts.append(generated_text)
			metadata.append(
				{
					"prompt_text": prompt_text,
					"sequence_index": int(sequence_index),
					"model_name": model_name,
				}
			)

	embeddings = extract_aligned_va_embeddings(texts, model_name, SELECTED_LAYERS)
	save_embeddings(embeddings, metadata, EMBEDDING_OUTPUT_DIR)