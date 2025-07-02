import os
from dotenv import load_dotenv
import streamlit as st

from core.rag_loader import load_rag_assistant
from interface.chat_handler import display_chat_history, handle_user_input
from interface.sidebar import setup_sidebar
from interface.styles.css_loader import load_css
from core.memory import get_memory


def main():
    """
    Launch the Streamlit RAG chatbot application.

    This function sets up environment variables, loads configurations, 
    initializes the RAG assistant and memory, applies CSS styling, 
    and starts the Streamlit UI with user input and chat memory support.

    Notes
    -----
    - Requires a `.env` file for environment variables.
    - Assumes vector DB and configs are already set up.
    - Loads user session with fixed ID `"default"`.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Disable tokenizer parallelism warning for HuggingFace models
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize RAG assistant and load app configuration
    rag, app_config = load_rag_assistant()

    # Apply custom CSS to Streamlit interface
    load_css("pet_style.css") 

    # Configure Streamlit app page
    st.set_page_config(page_title="🐶 Petshop Reviews - RAG Chatbot", page_icon="🦴")
    st.title("🐶 Petshop Reviews - RAG Chatbot")
    st.markdown("Ask questions based on real customer reviews from pet businesses.")

    # Use default session ID for chat history
    session_id = "default"

    # Load memory for the current session
    memory = get_memory(session_id)

    # Setup sidebar and retrieve user-defined parameters
    threshold, n_results = setup_sidebar(app_config, session_id)

    # Display previous chat messages (history)
    display_chat_history(memory)

    # Handle new user input and generate AI responses
    handle_user_input(rag, memory, threshold, n_results)


# Run the main app logic when executed directly
if __name__ == "__main__":
    main()
