"""
Builds a semantic vector database from processed customer reviews using ChromaDB.

This script performs the following steps:
- Loads a nested JSON file with place and review data.
- Extracts reviews and associated metadata.
- Splits review texts into token-based chunks.
- Generates sentence embeddings for each chunk using HuggingFace.
- Stores the resulting vectors and metadata in a ChromaDB persistent collection.

Note
----
Requires ChromaDB, LangChain, and a HuggingFace-supported embedding model.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import chromadb
import torch
from config.paths import JSON_PATH, VECTOR_DB_DIR
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import setup_logger

# Name of the ChromaDB collection used to store review embeddings
COLLECTION_NAME = "reviews"

# Initialize a logger for this module, writing to build_db.log
logger = setup_logger(name="build_db", log_filename="build_db.log")


def load_reviews_and_metadata(json_path: str) -> List[Tuple[str, Dict]]:
    """
    Load review texts and associated metadata from a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing places and reviews.

    Returns
    -------
    list of tuple
        A list of (review_text, metadata) tuples.
    """
    logger.info(f"Loading JSON from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        places = json.load(f)

    reviews_with_metadata = []
    for place in places:
        for review in place.get("reviews", []):
            text = review.get("text", "").strip()
            if not text:
                continue  # Skip empty reviews

            # Build metadata for each review
            metadata = {
                # Place fields
                "name": place.get("name"),
                "street": place.get("street"),
                "neighborhood": place.get("neighborhood"),
                "city": place.get("city"),
                "type": place.get("type"),
                "place_rating": place.get("rating"),

                # Review fields
                "review_rating": review.get("rating"),
                "author": review.get("author"),
                "date": review.get("date"),
                "response": review.get("response"),
            }

            # Store the review text along with its metadata    
            reviews_with_metadata.append((text, metadata))

    logger.info(f"Loaded {len(reviews_with_metadata)} reviews with metadata")
    return reviews_with_metadata


def initialize_db(persist_directory: str, collection_name: str, delete_existing: bool = False) -> chromadb.Collection:
    """
    Create or reset a ChromaDB collection.

    Parameters
    ----------
    persist_directory : str
        Path to store the ChromaDB collection.
    collection_name : str
        Name of the collection to use.
    delete_existing : bool, optional
        If True, deletes any existing data before creating the collection.

    Returns
    -------
    chromadb.Collection
        The initialized collection.
    """
    if os.path.exists(persist_directory) and delete_existing:
        logger.info(f"Deleting existing ChromaDB directory at {persist_directory}")

        # Remove existing database directory if requested
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Create a ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        # Try to load existing collection
        collection = client.get_collection(name=collection_name)
        logger.info(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If not found, create a new one using cosine similarity
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created new collection: {collection_name}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "reviews",
) -> chromadb.Collection:
    """
    Retrieve an existing ChromaDB collection.

    Parameters
    ----------
    persist_directory : str
        Path where ChromaDB data is stored.
    collection_name : str
        Name of the collection to retrieve.

    Returns
    -------
    chromadb.Collection
        The existing collection.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection(name=collection_name)


def chunk_reviews_by_tokens(
    reviews_with_metadata: List[Tuple[str, Dict]],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    encoding_name: str = "cl100k_base"
) -> List[Tuple[str, Dict]]:
    """
    Split review texts into chunks based on token count.

    Parameters
    ----------
    reviews_with_metadata : list of tuple
        List of (review_text, metadata) entries.
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of overlapping tokens between chunks.
    encoding_name : str
        Encoding used to count tokens (compatible with tiktoken).

    Returns
    -------
    list of tuple
        List of (chunk_text, metadata) entries with token-based splitting.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
    )

    chunked_reviews = []
    for review_text, metadata in reviews_with_metadata:
        # Split text into smaller overlapping chunks
        chunks = splitter.split_text(review_text)
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()         # Keep original metadata
            chunk_meta["chunk_index"] = i        # Add chunk index
            chunked_reviews.append((chunk, chunk_meta))

    logger.info(f"Generated {len(chunked_reviews)} chunks from {len(reviews_with_metadata)} reviews")
    return chunked_reviews


def embed_review_chunks(review_chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of review chunks.

    Parameters
    ----------
    review_chunks : list of str
        Review chunk texts.

    Returns
    -------
    list of list of float
        Embeddings represented as lists of floats.
    """
    # Automatically select best available device: CUDA > MPS > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Embedding chunks on device: {device}")

    # Load multilingual sentence embedding model
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        model_kwargs={"device": device},
    )

    # Generate vector representations for each chunk
    logger.info(f"Generated embeddings for {len(review_chunks)} chunks")
    return model.embed_documents(review_chunks)


def insert_review_chunks(
    collection: chromadb.Collection,
    chunked_reviews: List[Tuple[str, Dict]],
    batch_size: int = 5000
) -> None:
    """
    Insert review chunks and their embeddings into ChromaDB in batches.

    Parameters
    ----------
    collection : chromadb.Collection
        The ChromaDB collection to insert into.
    chunked_reviews : list of tuple
        List of (chunk_text, metadata) entries.
    batch_size : int
        Number of entries to insert per batch.

    Returns
    -------
    None
    """
    total = len(chunked_reviews)
    logger.info(f"Inserting {total} chunks into ChromaDB in batches of {batch_size}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunked_reviews[start:end]

        # Prepare data for insertion
        review_chunks = [text for text, _ in batch]
        metadatas = [meta for _, meta in batch]
        embeddings = embed_review_chunks(review_chunks)
        ids = [f"chunk_{i}" for i in range(start, end)]  # Generate unique IDs for each chunk

        # Add batch to ChromaDB
        collection.add(
            documents=review_chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(f"Inserted batch {start}–{end}")


def main() -> None:
    """
    Entry point of the script. Orchestrates the full embedding pipeline.

    Steps
    -----
    1. Load JSON data containing places and reviews.
    2. Extract and chunk review texts by token count.
    3. Generate embeddings using a HuggingFace model.
    4. Store chunked texts, embeddings, and metadata in a ChromaDB collection.
    """
    logger.info("Starting full embedding pipeline")

    # Load review texts and their metadata
    reviews_with_metadata = load_reviews_and_metadata(JSON_PATH)

    # Split reviews into token-based chunks
    chunked_reviews = chunk_reviews_by_tokens(reviews_with_metadata)

    # Initialize the ChromaDB collection (delete if already exists)
    collection = initialize_db(VECTOR_DB_DIR, COLLECTION_NAME, delete_existing=True)

    # Insert the chunks and embeddings into the vector database
    insert_review_chunks(collection, chunked_reviews)

    total = collection.count()
    logger.info(f"Pipeline complete: {total} documents in collection")


if __name__ == "__main__":
    # Run the main pipeline when script is executed directly
    main()