import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from settings import (
    SETTINGS_NAME,
    MODEL_NAME,
    GENERATED_OUTPUT_PATH,
    EMBEDDING_OUTPUT_DIR,
    HOVER_TEXT_EXPORT_FILENAME,
)


output_dir = Path(EMBEDDING_OUTPUT_DIR)


def main():
    generated = np.load(GENERATED_OUTPUT_PATH, allow_pickle=False)
    settings_data = json.loads(str(generated["settings_json"]))
    prompt_texts = generated["prompt_texts"]
    sequences = generated["sequences"]

    num_prompts, num_return, max_length = sequences.shape
    model_name = settings_data.get("model", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_texts = [[None for _ in range(num_return)] for _ in range(num_prompts)]
    char_end_offsets = np.zeros((num_prompts, num_return, max_length), dtype=np.int32)

    for prompt_index in range(num_prompts):
        for sequence_index in range(num_return):
            token_ids = sequences[prompt_index, sequence_index]
            full_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            full_texts[prompt_index][sequence_index] = full_text

            for token_index in range(max_length):
                prefix_text = tokenizer.decode(
                    token_ids[: token_index + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                char_end_offsets[prompt_index, sequence_index, token_index] = len(prefix_text)

        print(f"Exported hover text for prompt {prompt_index + 1}/{num_prompts}")

    output_path = output_dir / HOVER_TEXT_EXPORT_FILENAME
    np.savez_compressed(
        output_path,
        full_texts=np.array(full_texts, dtype=str),
        char_end_offsets=char_end_offsets,
        prompt_texts=np.array(prompt_texts, dtype=str),
        metadata_json=np.array(
            json.dumps(
                {
                    "settings_name": SETTINGS_NAME,
                    "model_name": model_name,
                    "num_prompts": int(num_prompts),
                    "num_return": int(num_return),
                    "max_length": int(max_length),
                    "skip_special_tokens": True,
                    "clean_up_tokenization_spaces": False,
                }
            )
        ),
    )

    print(f"Saved hover export to {output_path}")
    print(f"Shape: full_texts={np.array(full_texts, dtype=str).shape}, char_end_offsets={char_end_offsets.shape}")


if __name__ == "__main__":
    main()