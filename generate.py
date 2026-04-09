import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS
import torch
from settings import MODEL_NAME, MAX_LENGTH, TEMPERATURE, TOP_K, TOP_P, NUM_RETURN, GENERATED_OUTPUT_PATH


def generate_sequences(model_name, prompts, max_length, temperature, top_k, top_p, num_return):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt["text"], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            results.append({
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt_text": prompt["text"],
                "generated_text": text,
                "model": model_name,
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "sequence_index": i,
            })
        print(f"Generated {num_return} sequence(s) for prompt '{prompt['id']}'")

    return results


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} sequences to {output_path}")


if __name__ == "__main__":
    results = generate_sequences(
        model_name=MODEL_NAME,
        prompts=PROMPTS,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        num_return=NUM_RETURN,
    )
    save_results(results, GENERATED_OUTPUT_PATH)
