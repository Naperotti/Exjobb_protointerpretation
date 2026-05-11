SETTINGS_NAME = "Bank_prompts_40tokens_3000returns" #_MIN_SAMPLES20_MAX_EPS800
PROMPT_ID = "test_1"

MODEL_NAME = "Qwen/Qwen3-8B"
#models = ["gpt2", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]
MAX_LENGTH = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
NUM_RETURN = 100
GENERATED_OUTPUT_PATH = f"data/{SETTINGS_NAME}.npz"

EMBEDDING_INPUT_PATH = GENERATED_OUTPUT_PATH
EMBEDDING_MODE = "hidden"  # "hidden" or "sentence"
EMBEDDING_MODEL_NAME = None
EMBEDDING_LAYER = -1
EMBEDDING_OUTPUT_DIR = "embeddings"
EMBEDDING_ARRAY_FILENAME = f"aligned_va_embeddings_{SETTINGS_NAME}.npy"
EMBEDDING_METADATA_FILENAME = f"aligned_va_metadata_{SETTINGS_NAME}.json"

# Precompute output files (run-specific)
UMAP_PROJECTIONS_FILENAME = f"umap_projections_{SETTINGS_NAME}.npy"
OPTICS_LABELS_FILENAME = f"optics_labels_{SETTINGS_NAME}.npy"
OPTICS_REACHABILITY_FILENAME = f"optics_reachability_{SETTINGS_NAME}.npy"
OPTICS_ORDERINGS_FILENAME = f"optics_orderings_{SETTINGS_NAME}.npy"
OPTICS_METRICS_FILENAME = f"optics_metrics_{SETTINGS_NAME}.npz"