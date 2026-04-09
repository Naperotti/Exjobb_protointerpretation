MODEL_NAME = "Qwen/Qwen3-1.7B"
#models = ["gpt2", "Qwen/Qwen3-1.7B"]
MAX_LENGTH = 30
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
NUM_RETURN = 2
GENERATED_OUTPUT_PATH = "data/generated_sequences.json"

EMBEDDING_INPUT_PATH = GENERATED_OUTPUT_PATH
EMBEDDING_MODE = "hidden"  # "hidden" or "sentence"
EMBEDDING_MODEL_NAME = None
EMBEDDING_LAYER = -1
EMBEDDING_OUTPUT_DIR = "embeddings"