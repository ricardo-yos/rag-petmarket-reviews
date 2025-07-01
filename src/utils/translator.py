from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text: str) -> str:
    """
    Detect the language of the given input text.

    Parameters
    ----------
    text : str
        The input text whose language needs to be detected.

    Returns
    -------
    str
        The ISO 639-1 language code (e.g., 'en', 'pt', 'es').
        Returns "unknown" if detection fails.

    Notes
    -----
    - Uses the `langdetect` library for language identification.
    - May be inaccurate for very short or ambiguous text.
    """
    try:
        return detect(text)
    except:
        # Return a fallback value if detection fails
        return "unknown"

def translate(text: str, target_lang: str) -> str:
    """
    Translate the given text into the specified target language.

    Parameters
    ----------
    text : str
        The input text to translate.
    target_lang : str
        The target language code (e.g., 'en' for English, 'pt' for Portuguese).

    Returns
    -------
    str
        The translated text in the target language.

    Notes
    -----
    - Uses GoogleTranslator from `deep_translator` with auto-detection of the source language.
    - Requires internet connection to function properly.
    """
    return GoogleTranslator(source='auto', target=target_lang).translate(text)