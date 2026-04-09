import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from settings import (
    EMBEDDING_INPUT_PATH,
    EMBEDDING_MODE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_LAYER,
    EMBEDDING_OUTPUT_DIR,
)


def load_sequences(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_hidden_states(texts, model_name, layer=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.hidden_states[layer]
        # mean pooling over token dimension
        embedding = hidden.squeeze(0).mean(dim=0).numpy()
        embeddings.append(embedding)
        print(f"Extracted hidden states for sequence {i+1}/{len(texts)}")

    return np.array(embeddings)


def extract_sentence_transformer(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # mean pooling over tokens, ignoring padding
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
        embeddings.append(embedding.squeeze(0).numpy())
        print(f"Extracted sentence embedding for sequence {i+1}/{len(texts)}")

    return np.array(embeddings)


def save_embeddings(embeddings, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved embeddings {embeddings.shape} to {output_dir}")


if __name__ == "__main__":
    sequences = load_sequences(EMBEDDING_INPUT_PATH)
    texts = [s["generated_text"] for s in sequences]
    metadata = [{"prompt_id": s["prompt_id"], "category": s["category"], "prompt_text": s["prompt_text"]} for s in sequences]

    if EMBEDDING_MODE == "hidden":
        model_name = EMBEDDING_MODEL_NAME or sequences[0]["model"]
        embeddings = extract_hidden_states(texts, model_name, layer=EMBEDDING_LAYER)
    else:
        model_name = EMBEDDING_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = extract_sentence_transformer(texts, model_name)

    save_embeddings(embeddings, metadata, EMBEDDING_OUTPUT_DIR)
