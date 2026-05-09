SETTINGS_NAME = "Bank_prompts_40tokens_500returns1"
PROMPT_ID = "test_1"

MODEL_NAME = "Qwen/Qwen3-8B"
#models = ["gpt2", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]
MAX_LENGTH = 40
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
NUM_RETURN = 500
GENERATED_OUTPUT_PATH = f"data/{SETTINGS_NAME}.npz"

EMBEDDING_INPUT_PATH = GENERATED_OUTPUT_PATH
EMBEDDING_MODE = "hidden"  # "hidden" or "sentence"
EMBEDDING_MODEL_NAME = None
EMBEDDING_LAYER = -1
EMBEDDING_OUTPUT_DIR = "embeddings"
EMBEDDING_ARRAY_FILENAME = "aligned_va_embeddings_40tokens_500returns1.npy"
EMBEDDING_METADATA_FILENAME = "aligned_va_metadata_40tokens_500returns1.json"