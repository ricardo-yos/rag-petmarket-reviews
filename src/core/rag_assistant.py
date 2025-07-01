"""
rag_assistant.py

Module implementing the RAGAssistant class for retrieval-augmented generation
using vector search, prompt building, persistent memory, and LLM interaction.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from utils.logger import setup_logger
from utils.translator import detect_language, translate
from data_processing.build_db import get_db_collection, embed_review_chunks
from .prompt_builder import build_prompt_from_config
from .memory import get_memory
from .llm_helpers import (
    count_tokens,
    needs_history_context,
    fix_markdown_response,
)

# Setup logger for this module writing to rag_assistant.log
logger = setup_logger(name="rag_assistant", log_filename="rag_assistant.log")


class RAGAssistant:
    """
    Retrieval-Augmented Generation (RAG) assistant that handles vector retrieval,
    prompt construction, persistent conversation memory, and LLM interaction.

    Attributes
    ----------
    collection : chromadb.Collection
        Vector database collection to query for relevant documents.
    embed_func : Callable
        Embedding function to generate embeddings for queries.
    prompt_config : dict
        Configuration dictionary for prompt templates loaded from YAML.
    app_config : dict
        General application configuration loaded from YAML.
    model_name : str
        Identifier of the LLM model to use.
    memory : ConversationBufferMemory
        Conversation memory for storing past messages.
    llm_client : Any
        LLM client instance for generating completions.
    """

    def __init__(self, collection, embed_func, prompt_config, app_config, model_name, session_id="default"):
        """
        Initialize a RAGAssistant instance with its dependencies.

        Parameters
        ----------
        collection : chromadb.Collection
            Vector database collection.
        embed_func : Callable
            Embedding function for queries.
        prompt_config : dict
            Dictionary defining prompt templates and structure.
        app_config : dict
            General application settings and thresholds.
        model_name : str
            Name of the language model to use (e.g., "gpt-4").
        session_id : str, optional
            Identifier for chat session memory (default is "default").
        """
        self.collection = collection
        self.embed_func = embed_func
        self.prompt_config = prompt_config
        self.app_config = app_config
        self.model_name = model_name
        self.llm_client = ChatGroq(model=model_name)
        self.memory = get_memory(session_id)

        logger.info(f"RAGAssistant initialized with model '{model_name}' and session '{session_id}'")


    def format_review(self, doc: str, meta: dict) -> str:
        """
        Format a review document with light metadata for better context.

        Parameters
        ----------
        doc : str
            Review text content.
        meta : dict
            Metadata associated with the review.

        Returns
        -------
        str
            Formatted string with metadata and review content.
        """
        name = meta.get("name", "Unknown name")
        rating = meta.get("place_rating", "N/A")
        street = meta.get("street", "No street provided")
        neighborhood = meta.get("neighborhood", "No neighborhood provided")
        city = meta.get("city", "No city provided")

        return (
            f"{name} (Rating: {rating}) — {street}, {neighborhood}, {city}\n"
            f"Review: {doc.strip()}"
        )


    def retrieve_relevant_reviews(self, query: str, n_results: int = 5, threshold: float = 0.3) -> list[str]:
        """
        Retrieve the top-N most relevant reviews based on query similarity.

        Parameters
        ----------
        query : str
            User's input query.
        n_results : int, optional
            Number of top documents to return (default is 5).
        threshold : float, optional
            Cosine distance threshold for filtering results (default is 0.3).

        Returns
        -------
        list[str]
            List of formatted review strings.
        """
        logger.info(f"Retrieving top {n_results} similar documents for query: '{query}'")

        try:
            # Generate vector embedding for the query
            query_embedding = self.embed_func([query])[0]

            # Query the vector DB for similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}", exc_info=True)
            return []

        # Check if any documents returned
        if not results.get("documents") or not results["documents"][0]:
            logger.warning("No documents returned from vector search.")
            return []

        # Filter documents based on distance threshold
        relevant = []
        for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
            if dist < threshold:
                formatted = self.format_review(doc, meta)
                relevant.append(formatted)

        logger.debug(f"Found {len(relevant)} relevant documents under threshold {threshold}.")
        return relevant


    def respond(self, query: str, n_results: int = 5, threshold: float = 0.3) -> str:
        """
        Generate a natural language response to the user's query using retrieved documents,
        persistent memory, and an LLM.

        This method orchestrates the full Retrieval-Augmented Generation (RAG) pipeline:
        1. Detects the language of the user input.
        2. Retrieves relevant review documents from a vector store based on similarity to the query.
        3. Loads past conversation history from memory.
        4. Optionally summarizes the history if it exceeds token limits.
        5. Determines whether chat history is required for proper understanding.
        6. Builds a context-enriched prompt using configuration templates.
        7. Sends the prompt to the LLM and obtains a response.
        8. Optionally translates the response to match the user's input language.
        9. Formats the response with corrected Markdown syntax.

        Parameters
        ----------
        query : str
            The user's input question or prompt.
        n_results : int, optional
            Number of similar documents to retrieve from the vector database (default is 5).
        threshold : float, optional
            Maximum cosine distance for considering a document relevant (default is 0.3).

        Returns
        -------
        str
            The final language model response, translated and formatted if necessary.

        Notes
        -----
        - Uses persistent memory to maintain context across user turns.
        - Handles multilingual queries and auto-translates the result when needed.
        """
        logger.info(f"Generating response for query: '{query}'")

        # Detect language of the input
        detected_lang = detect_language(query)
        logger.info(f"Detected language: {detected_lang}")

        # Load vector retrieval config from app_config
        threshold = self.app_config.get("vectordb", {}).get("threshold", 0.3)
        n_results = self.app_config.get("vectordb", {}).get("n_results", 5)
        
        # Retrieve relevant reviews
        relevant_reviews = self.retrieve_relevant_reviews(query, n_results, threshold)

        # Load chat memory and apply trimming strategy
        past_messages = self.memory.load_memory_variables({}).get("chat_history", [])
        window_size = self.app_config.get("memory_strategies", {}).get("trimming_window_size", 6)
        chat_history = past_messages[-(window_size * 2):]  # Keep last N Q&A pairs
        logger.debug(f"Loaded {len(chat_history)} messages from memory (window size: {window_size}).")

        # Summarize if token count exceeds the maximum
        max_tokens = self.app_config.get("memory_strategies", {}).get("summarization_max_tokens", 1000)
        chat_text = "\n".join(msg.content for msg in chat_history)
        total_tokens = count_tokens(chat_text, model_name=self.model_name)

        if total_tokens > max_tokens:
            logger.info(f"Chat history exceeds {max_tokens} tokens ({total_tokens} tokens). Summarizing...")

            summarization_prompt = f"""
            Summarize the following chat history to preserve useful context for the next user query.
            Be concise, accurate, and preserve the intent of both questions and answers.

            Chat history:
            {chat_text}
            """

            try:
                summary_response = self.llm_client.invoke([HumanMessage(content=summarization_prompt)])
                summary_text = summary_response.content.strip()
                logger.info("Chat history summarized successfully.")
                chat_history = [HumanMessage(content="Summary of previous conversation:\n" + summary_text)]
            except Exception as e:
                logger.warning(f"Failed to summarize chat history: {e}", exc_info=True)

        # Build context string from last Q&A pair for reasoning
        last_turn = "\n".join(msg.content for msg in past_messages[-2:]) if len(past_messages) >= 2 else ""
        use_history = needs_history_context(query, last_turn, self.llm_client) if last_turn else False

        # Context string passed to prompt (if needed)
        if use_history:
            context = (
                f"{chr(10).join(relevant_reviews)}\n\n"
                f"Histórico:\n{last_turn}\n\n"
                f"Nova pergunta:\n{query}"
            )
        else:
            context = (
                f"{chr(10).join(relevant_reviews)}\n\n"
                f"Pergunta:\n{query}"
            )        

        # Get default reasoning strategy name from config (e.g., "CoT")
        strategy_name = self.app_config.get("reasoning_strategies", {}).get("default", "CoT")
        
        # Get reasoning instruction text for the chosen strategy
        reasoning_instruction = self.app_config.get("reasoning_strategies", {}).get(strategy_name, "")

        # Build a prompt that combines retrieved reviews, query, and reasoning strategy
        prompt = build_prompt_from_config(
            config=self.prompt_config,
            documents=relevant_reviews,
            query=query,
            app_config=self.app_config,
            reasoning_instruction=reasoning_instruction
        )

        # Send the constructed prompt to the LLM for response generation
        full_messages = past_messages + [HumanMessage(content=prompt)]
        logger.info("Sending context-enriched prompt to LLM")
        response = self.llm_client.invoke(full_messages)

        # Optional translation
        if detected_lang != "pt":
            translated_response = translate(response.content, target_lang=detected_lang)
            formatted_response = fix_markdown_response(translated_response, self.llm_client)
            logger.info(f"Translated response to: {detected_lang}")
            return formatted_response

        # Return raw LLM output if language is Portuguese
        return response.content