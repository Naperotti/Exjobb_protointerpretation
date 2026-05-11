import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


# Chunked run configuration
TOTAL_RETURNS = 500
CHUNK_RETURNS = 100
BASE_SETTINGS_NAME = "Bank_prompts_200tokens_500returns"


ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "settings.py"
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"


def replace_setting(text, key, value_literal):
    pattern = rf"(?m)^{key}\s*=\s*.*$"
    replacement = f"{key} = {value_literal}"
    new_text, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise RuntimeError(f"Could not update setting '{key}' in settings.py")
    return new_text


def write_chunk_settings(chunk_idx, chunk_returns):
    settings_name = f"{BASE_SETTINGS_NAME}_part{chunk_idx}"
    emb_array_name = f"aligned_va_embeddings_{BASE_SETTINGS_NAME}_part{chunk_idx}.npy"
    emb_meta_name = f"aligned_va_metadata_{BASE_SETTINGS_NAME}_part{chunk_idx}.json"

    text = SETTINGS_PATH.read_text(encoding="utf-8")
    text = replace_setting(text, "SETTINGS_NAME", json.dumps(settings_name))
    text = replace_setting(text, "NUM_RETURN", str(chunk_returns))
    text = replace_setting(text, "EMBEDDING_ARRAY_FILENAME", json.dumps(emb_array_name))
    text = replace_setting(text, "EMBEDDING_METADATA_FILENAME", json.dumps(emb_meta_name))
    SETTINGS_PATH.write_text(text, encoding="utf-8")

    return {
        "settings_name": settings_name,
        "npz_path": DATA_DIR / f"{settings_name}.npz",
        "emb_path": EMBED_DIR / emb_array_name,
        "meta_path": EMBED_DIR / emb_meta_name,
        "chunk_returns": chunk_returns,
    }


def run_cmd(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def merge_npz_files(chunk_infos):
    npz_items = [np.load(info["npz_path"], allow_pickle=False) for info in chunk_infos]

    prompt_texts = npz_items[0]["prompt_texts"]
    settings_data = json.loads(str(npz_items[0]["settings_json"]))

    seqs = [item["sequences"] for item in npz_items]
    ents = [item["entropies"] for item in npz_items]

    merged_sequences = np.concatenate(seqs, axis=1)
    merged_entropies = np.concatenate(ents, axis=1)

    settings_data["settings_name"] = BASE_SETTINGS_NAME
    settings_data["num_return"] = int(merged_sequences.shape[1])

    out_path = DATA_DIR / f"{BASE_SETTINGS_NAME}.npz"
    np.savez_compressed(
        out_path,
        settings_json=np.array(json.dumps(settings_data)),
        prompt_texts=np.array(prompt_texts, dtype=str),
        sequences=np.array(merged_sequences, dtype=np.int64),
        entropies=np.array(merged_entropies, dtype=np.float32),
    )
    print(f"Merged NPZ saved to {out_path}")


def merge_embedding_files(chunk_infos):
    emb_arrays = [np.load(info["emb_path"]) for info in chunk_infos]
    merged_embeddings = np.concatenate(emb_arrays, axis=0)

    merged_metadata = []
    max_index_by_prompt = {}

    for info in chunk_infos:
        meta = json.loads(info["meta_path"].read_text(encoding="utf-8"))
        for row in meta:
            prompt = row["prompt_text"]
            cur_max = max_index_by_prompt.get(prompt, 0)
            new_row = dict(row)
            new_row["sequence_index"] = int(row["sequence_index"]) + cur_max
            merged_metadata.append(new_row)

        # Update max counters per prompt using this chunk's metadata
        local_max = {}
        for row in meta:
            prompt = row["prompt_text"]
            local_max[prompt] = max(local_max.get(prompt, 0), int(row["sequence_index"]))
        for prompt, m in local_max.items():
            max_index_by_prompt[prompt] = max_index_by_prompt.get(prompt, 0) + m

    out_emb = EMBED_DIR / f"aligned_va_embeddings_{BASE_SETTINGS_NAME}.npy"
    out_meta = EMBED_DIR / f"aligned_va_metadata_{BASE_SETTINGS_NAME}.json"

    np.save(out_emb, merged_embeddings.astype(np.float32))
    out_meta.write_text(json.dumps(merged_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Merged embeddings saved to {out_emb}")
    print(f"Merged metadata saved to {out_meta}")


def main():
    original_settings = SETTINGS_PATH.read_text(encoding="utf-8")
    chunk_infos = []

    try:
        chunk_count = math.ceil(TOTAL_RETURNS / CHUNK_RETURNS)
        for i in range(chunk_count):
            remaining = TOTAL_RETURNS - (i * CHUNK_RETURNS)
            chunk_returns = CHUNK_RETURNS if remaining >= CHUNK_RETURNS else remaining
            info = write_chunk_settings(i + 1, chunk_returns)
            chunk_infos.append(info)

            print(f"\n=== Chunk {i + 1}/{chunk_count}: {chunk_returns} returns ===", flush=True)
            run_cmd([sys.executable, "generate.py"])
            run_cmd([sys.executable, "aligned_VA_embeddings.py"])

        print("\n=== Merging chunk outputs ===", flush=True)
        merge_npz_files(chunk_infos)
        merge_embedding_files(chunk_infos)
        print("Done.")

    finally:
        SETTINGS_PATH.write_text(original_settings, encoding="utf-8")
        print("settings.py restored.")


if __name__ == "__main__":
    main()
