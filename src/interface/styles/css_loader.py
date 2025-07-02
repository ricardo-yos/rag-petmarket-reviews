import os
import streamlit as st


def load_css(relative_path: str) -> None:
    """
    Load and apply custom CSS styling to the Streamlit app.

    This function reads a CSS file from the specified relative path
    (relative to this Python file's location) and injects its content
    into the Streamlit app using Markdown and the `<style>` tag.

    Parameters
    ----------
    relative_path : str
        Relative path to the CSS file, relative to the location of this script.

    Notes
    -----
    - The file must be a valid CSS file.
    - `unsafe_allow_html=True` is required for Streamlit to render custom styles.
    """
    # Resolve the absolute path of the CSS file based on this script's location
    css_path = os.path.join(os.path.dirname(__file__), relative_path)

    # Read the content of the CSS file
    with open(css_path, "r") as f:
        css = f.read()

    # Inject the CSS into the Streamlit app using HTML
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)