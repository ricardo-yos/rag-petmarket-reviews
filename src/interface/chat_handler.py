import streamlit as st
from utils.logger import setup_logger

# Initialize the logger for the application
logger = setup_logger(name="app", log_filename="app.log")


def display_chat_history(memory) -> None:
    """
    Display previous chat history messages in the chat UI.

    This function retrieves past messages stored in the session memory
    and renders them as chat bubbles using Streamlit's chat interface.

    Parameters
    ----------
    memory : ConversationBufferMemory
        Memory object containing the chat history.
    """
    # Load the full chat history for the session
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Display each message with appropriate avatar
    for msg in chat_history:
        role = msg.type
        content = msg.content
        avatar = "🐕" if role in ["user", "human"] else "💬"
        st.chat_message(role, avatar=avatar).markdown(content)


def handle_user_input(rag, memory, threshold: float, n_results: int) -> None:
    """
    Handle user input, generate response using RAG, and update chat memory.

    This function waits for user input, uses the RAG assistant to generate
    a response based on semantic search, updates the conversation memory,
    and triggers a rerun of the app to display the updated chat.

    Parameters
    ----------
    rag : RAGAssistant
        The RAG pipeline assistant instance, responsible for responding to queries.
    memory : ConversationBufferMemory
        Chat memory object used to store user and assistant messages.
    threshold : float
        Cosine similarity threshold for retrieving relevant reviews.
    n_results : int
        Number of top documents (reviews) to retrieve for the response.

    Notes
    -----
    - If the user input is empty, the function does nothing.
    - Any exceptions during response generation are caught and logged.
    """
    # Wait for user input via the Streamlit chat input box
    if user_input := st.chat_input("Ask something about petshop reviews:"):
        logger.info(f"User question: {user_input}")

        with st.spinner("Thinking... 💭"):
            try:
                # Generate assistant response using RAG pipeline
                response = rag.respond(
                    query=user_input,
                    n_results=n_results,
                    threshold=threshold
                )
                logger.info("Response successfully generated.")
            except Exception as e:
                # Log and handle any errors during response generation
                logger.error(f"Error generating response: {e}", exc_info=True)
                response = "Sorry, an error occurred while processing your question."

        # Save messages to memory for chat continuity
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)

        # Rerun the app to show the updated chat state
        st.rerun()