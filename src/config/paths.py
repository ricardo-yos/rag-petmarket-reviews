import os

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Top-level directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")

# Data subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")                    # Ex: places.csv, reviews.csv
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")        # Ex: places_reviews.json
VECTOR_DB_DIR = os.path.join(DATA_DIR, "chroma")                # ChromaDB SQLite

# Files
CHAT_HISTORY_DB_FPATH = os.path.join(MEMORY_DIR, "chat_history.db")
JSON_PATH = os.path.join(PROCESSED_DATA_DIR, "places_reviews.json")

# Config files
ENV_PATH = os.path.join(ROOT_DIR, ".env")
APP_CONFIG_FPATH = os.path.join(ROOT_DIR, "src", "config", "app_config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "src", "config", "prompt_config.yaml")
