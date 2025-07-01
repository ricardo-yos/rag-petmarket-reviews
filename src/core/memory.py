import sqlite3
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from config.paths import CHAT_HISTORY_DB_FPATH
from utils.logger import setup_logger

# Create a logger for the "memory" module that writes to "memory.log"
logger = setup_logger(name="memory", log_filename="memory.log")


def ensure_chat_table():
    """
    Ensure the 'message_store' table exists in the SQLite database.

    This function creates the `message_store` table if it does not already exist.
    It supports persistent chat memory by storing messages for each session.

    Notes
    -----
    - Uses the database file path defined in `CHAT_HISTORY_DB_FPATH`.
    - The table stores session_id, message content, and timestamp.
    - Logs success or failure of the operation.

    Raises
    ------
    Exception
        If an error occurs during database connection or table creation.
    """
    try:
        logger.debug(f"Connecting to SQLite DB at {CHAT_HISTORY_DB_FPATH} to ensure chat table exists.")
        conn = sqlite3.connect(CHAT_HISTORY_DB_FPATH)  # Connect to SQLite database
        cursor = conn.cursor()

        # Create the 'message_store' table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

        logger.info("Table 'message_store' checked or created successfully.")
    except Exception as e:
        logger.exception(f"Failed to check/create 'message_store' table: {e}")
        raise


def get_memory(session_id: str = "default") -> ConversationBufferMemory:
    """
    Initialize and return a persistent conversation memory object for a given session.

    This function ensures the message table exists, then sets up SQL-based
    chat history for persistent storage using LangChain's SQLChatMessageHistory.
    The conversation memory buffers chat messages and persists them across sessions.

    Parameters
    ----------
    session_id : str, optional
        Identifier for the chat session (default is "default").

    Returns
    -------
    ConversationBufferMemory
        A LangChain memory object that manages chat history with persistent storage.

    Notes
    -----
    - Relies on SQLite database file at `CHAT_HISTORY_DB_FPATH`.
    - Uses SQLChatMessageHistory to interface with the database.
    """
    logger.debug(f"Initializing persistent memory for session_id: '{session_id}'")

    ensure_chat_table()  # Make sure the chat message table exists

    # Create SQLChatMessageHistory tied to the session and database connection
    history = SQLChatMessageHistory(
        connection=f"sqlite:///{CHAT_HISTORY_DB_FPATH}",
        session_id=session_id,
    )

    # Wrap SQL chat history in ConversationBufferMemory for LangChain usage
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=history,
        return_messages=True,
    )

    logger.info(f"Persistent memory initialized for session_id: '{session_id}'")

    return memory