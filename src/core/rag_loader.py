from core.rag_assistant import RAGAssistant
from data_processing.build_db import get_db_collection, embed_review_chunks
from config.config_loader import load_yaml_config
from config.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH
from utils.logger import setup_logger

# Initialize app logger writing to "app.log"
logger = setup_logger(name="app", log_filename="app.log")


def load_rag_assistant():
    """
    Load and initialize the Retrieval-Augmented Generation (RAG) assistant.

    This function performs the following steps:
    - Loads the vector database collection containing review embeddings.
    - Loads application and prompt configurations from YAML files.
    - Extracts model configuration and prompt template.
    - Initializes a RAGAssistant instance with the loaded components.

    Returns
    -------
    rag : RAGAssistant
        The initialized RAGAssistant instance ready to respond to queries.
    app_config : dict
        The loaded application configuration dictionary.

    Notes
    -----
    - Assumes a ChromaDB instance with a collection named "reviews" exists.
    - Requires valid YAML config files at the specified APP_CONFIG_FPATH and PROMPT_CONFIG_FPATH.
    - The embedding function `embed_review_chunks` must be defined and compatible.
    """
    logger.info("Loading RAGAssistant and configurations.")

    # Load vector DB collection named "reviews"
    collection = get_db_collection(collection_name="reviews")

    # Load application-level configuration settings from YAML
    app_config = load_yaml_config(APP_CONFIG_FPATH)

     # Load prompt configuration and extract RAG prompt settings
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
    rag_prompt_config = prompt_config["rag_assistant_prompt"]

    # Extract model name (e.g., "gpt-4") from app config
    llm_model = app_config["llm"]

    # Initialize RAGAssistant with collection, embedding, prompt, and model details
    rag = RAGAssistant(
        collection=collection,
        embed_func=embed_review_chunks,
        prompt_config=rag_prompt_config,
        app_config=app_config,
        model_name=llm_model,
    )

    logger.info("RAGAssistant initialized successfully.")
    return rag, app_config
