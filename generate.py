import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS
import torch
from settings import SETTINGS_NAME, PROMPT_ID, MODEL_NAME, MAX_LENGTH, TEMPERATURE, TOP_K, TOP_P, NUM_RETURN, GENERATED_OUTPUT_PATH


# Generates token sequences and per-token entropy for each prompt.
# Returns a list of dicts with keys: "prompt", "sequences", "entropies".
def generate_sequences(model_name, prompts, max_length, temperature, top_k, top_p, num_return):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in prompts:
        # inputs["input_ids"]: int64 tensor, shape [1, prompt_len]
        inputs = tokenizer(prompt["text"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            # outputs.sequences: int64 tensor, shape [num_return, prompt_len + max_length]
            # outputs.scores: tuple of max_length tensors, each shape [num_return, vocab_size]
            #   scores are raw logits before sampling (pre-temperature already applied by generate)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Compute Shannon entropy H = -sum(p * log(p)) at each decoding step.
        # scores already have top-k/top-p applied: filtered tokens are set to -inf by generate.
        # We filter to finite values per row before softmax to avoid computing over ~150k zeros.
        # Each step_scores: float32 tensor, shape [num_return, vocab_size]
        step_entropies = []
        for step_scores in outputs.scores:
            # Process each sequence separately because top-p cuts off at different points per row
            ent = torch.zeros(step_scores.shape[0], device=step_scores.device)
            for j in range(step_scores.shape[0]):
                row = step_scores[j]                     # [vocab_size]
                finite = row[row.isfinite()]             # [~top_k or fewer after top-p]
                probs = torch.softmax(finite, dim=0)
                log_probs = torch.log_softmax(finite, dim=0)
                ent[j] = -(probs * log_probs).sum()     # scalar
            step_entropies.append(ent)                   # [num_return]

        # Stack over time steps then transpose: [max_length, num_return] -> [num_return, max_length]
        # Final shape: [num_return, max_length], dtype float32
        entropies = torch.stack(step_entropies, dim=0).transpose(0, 1).cpu().numpy()

        prompt_result = {
            "prompt": prompt["text"],
            "sequences": {},  # token id arrays, each shape [max_length], dtype int64
            "entropies": {},  # entropy arrays, each shape [max_length], dtype float32
        }
        prompt_len = inputs["input_ids"].shape[1]
        for i, output in enumerate(outputs.sequences):
            # Slice off the prompt tokens, keep only generated tokens
            generated_ids = output[prompt_len:].cpu().numpy()  # [max_length], int64
            prompt_result["sequences"][str(i + 1)] = generated_ids
            prompt_result["entropies"][str(i + 1)] = entropies[i]
        results.append(prompt_result)
        print(f"Generated {num_return} sequence(s) for prompt '{prompt['id']}'")

    return results


# Packs results into a compressed .npz file alongside generation settings.
def save_results(results, output_path):
    settings_data = {
        "settings_name": SETTINGS_NAME,
        "prompt_id": PROMPT_ID,
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "num_return": NUM_RETURN,
    }

    prompt_texts = []
    sequences_2d = []
    entropies_2d = []

    for prompt_result in results:
        prompt_texts.append(prompt_result["prompt"])
        seqs = []
        ents = []
        for i in range(len(prompt_result["sequences"])):
            seqs.append(prompt_result["sequences"][str(i + 1)])
            ents.append(prompt_result["entropies"][str(i + 1)])
        sequences_2d.append(seqs)
        entropies_2d.append(ents)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Saved arrays:
    #   settings_json : scalar string (JSON)
    #   prompt_texts  : [num_prompts], str
    #   sequences     : [num_prompts, num_return, max_length], int64
    #   entropies     : [num_prompts, num_return, max_length], float32
    np.savez_compressed(
        output_path,
        settings_json=np.array(json.dumps(settings_data)),
        prompt_texts=np.array(prompt_texts, dtype=str),
        sequences=np.array(sequences_2d, dtype=np.int64),
        entropies=np.array(entropies_2d, dtype=np.float32),
    )
    print(f"Saved {len(results)} prompt groups to {output_path}")


# Loads the .npz and prints each array's shape as a quick sanity check.
def print_npz_shapes(output_path):
    data = np.load(output_path, allow_pickle=False)
    print("NPZ sanity check:")
    print(f"- settings_json shape: {data['settings_json'].shape}")
    print(f"- prompt_texts shape: {data['prompt_texts'].shape}")
    print(f"- sequences shape: {data['sequences'].shape}")
    print(f"- entropies shape: {data['entropies'].shape}")


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
    print_npz_shapes(GENERATED_OUTPUT_PATH)
