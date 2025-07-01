import tiktoken
from langchain_core.messages import HumanMessage
from utils.logger import setup_logger

# Initialize logger for LLM helpers
logger = setup_logger(name="llm_utils", log_filename="llm_utils.log")


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count the number of tokens in a given text for a specific model.

    Parameters
    ----------
    text : str
        The input text to be tokenized.
    model_name : str, optional
        Model name used to select the tokenizer (default is "gpt-4").

    Returns
    -------
    int
        Total number of tokens in the text.

    Notes
    -----
    Falls back to the 'cl100k_base' encoding if the model is not recognized.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)  # Select tokenizer for the model
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    tokens = encoding.encode(text)  # Tokenize the input
    return len(tokens)


def needs_history_context(query: str, last_turn: str, llm_client) -> bool:
    """
    Determine whether the current user query depends on previous chat history.

    Parameters
    ----------
    query : str
        The user's current question.
    last_turn : str
        The last Q/A pair from the conversation.
    llm_client : Any
        The language model client used to evaluate the dependency.

    Returns
    -------
    bool
        True if the current query requires prior context, False otherwise.

    Notes
    -----
    Sends a prompt asking the LLM if history is needed and expects "SIM" or "NÃO" in response.
    """
    check_prompt = f"""
    O usuário fez a pergunta: "{query}"

    Ela depende do seguinte histórico para ser compreendida?
    Histórico:
    "{last_turn}"

    Responda apenas com "SIM" ou "NÃO".
    """

    try:
        response = llm_client.invoke([HumanMessage(content=check_prompt)])
        return "sim" in response.content.strip().lower()  # Normalize response and check
    except Exception as e:
        logger.warning(f"Error checking history dependency: {e}", exc_info=True)
        return False


def fix_markdown_response(raw_response: str, llm_client) -> str:
    """
    Improve the Markdown formatting of an LLM-generated response.

    Parameters
    ----------
    raw_response : str
        The raw text output from the LLM.
    llm_client : Any
        The LLM client used to perform formatting correction.

    Returns
    -------
    str
        The cleaned response, properly formatted in Markdown.

    Notes
    -----
    - Keeps original wording and meaning unchanged.
    - Converts list-like content into proper bullet points.
    - Improves line breaks and readability using Markdown conventions.
    """
    prompt = (
        "Correct the Markdown formatting of the following text. "
        "Do not change the wording, structure, or meaning. "
        "If you detect multiple items listed in the text, convert them into bullet points with '-' markers. "
        "Only fix Markdown syntax issues and improve readability:\n\n"
        f"{raw_response}"
    )

    try:
        messages = [HumanMessage(content=prompt)]
        response = llm_client.invoke(messages)
        return response.content
    except Exception as e:
        logger.warning(f"Error formatting Markdown with LLM: {e}", exc_info=True)
        return raw_response  # Return original if formatting fails