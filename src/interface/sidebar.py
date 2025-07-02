import streamlit as st
import sqlite3
from config.paths import CHAT_HISTORY_DB_FPATH
from utils.logger import setup_logger

# Initialize logger for app events
logger = setup_logger(name="app", log_filename="app.log")


def setup_sidebar(app_config: dict, session_id: str) -> tuple[float, int]:
    """
    Create sidebar widgets for user configuration and chat management.

    This function builds the sidebar in the Streamlit app UI, where the user can:
    - Adjust the cosine similarity threshold for document retrieval
    - Set the number of top documents (reviews) to retrieve
    - Clear the chat history for the current session

    Parameters
    ----------
    app_config : dict
        Application configuration dictionary loaded from YAML.
    session_id : str
        Unique identifier for the current user session.

    Returns
    -------
    threshold : float
        Cosine similarity threshold value selected by the user.
    n_results : int
        Number of top relevant documents (reviews) to retrieve.

    Notes
    -----
    - When "Clear Chat History" is clicked, this function will erase all previous
      messages for the current session and restart the app.
    """
    with st.sidebar:
        st.header("Settings")

        # Input: cosine similarity threshold for semantic search
        threshold = st.number_input(
            "Cosine Distance Threshold",
            value=app_config["vectordb"]["threshold"],
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )

        # Input: number of top reviews to retrieve
        n_results = st.number_input(
            "Top K reviews to retrieve",
            value=app_config["vectordb"]["n_results"],
            min_value=1,
            max_value=100,
            step=1,
        )

        # Button to clear chat history
        if st.button("Clear Chat History"):
            clear_chat_history(session_id)
            st.rerun()

        # Footer section
        st.markdown("---")
        st.caption("""
            **ℹ️ About this app**

            This assistant retrieves and analyzes customer reviews from pet-related businesses using semantic search and a custom RAG pipeline.  
            
            Built by Ricardo Yoshitomi 🐾
        """)

    return threshold, n_results


def clear_chat_history(session_id: str) -> None:
    """
    Delete chat history for a given session from the SQLite message store.

    This function removes all stored chat messages from the local SQLite database
    for the specified session ID. It is used when the user clicks "Clear Chat History"
    in the Streamlit sidebar.

    Parameters
    ----------
    session_id : str
        The session identifier whose chat history should be deleted.

    Notes
    -----
    - The database path is configured in `CHAT_HISTORY_DB_FPATH`.
    - The table `message_store` must exist with a `session_id` column.
    """
    # Connect to local SQLite DB
    conn = sqlite3.connect(CHAT_HISTORY_DB_FPATH)
    cursor = conn.cursor()

    # Delete all messages for the given session
    cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

    logger.info(f"User cleared the chat history for session: {session_id}")