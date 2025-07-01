from typing import Dict, Any, Optional


def format_prompt_section(title: str, content: any) -> str:
    """
    Format a titled section of the prompt for better readability.

    Parameters
    ----------
    title : str
        The section title (e.g., "Constraints:")
    content : any
        The content under the section. Can be a list, dict or str.

    Returns
    -------
    str
        Formatted string representing the section.
    """
    if isinstance(content, list):
        return f"{title}\n" + "\n".join(f"- {item}" for item in content)
    elif isinstance(content, dict):
        return f"{title}\n" + "\n".join(f"- {key}: {value}" for key, value in content.items())
    else:
        return f"{title}\n- {content}"


def build_prompt_from_config(config, documents, query, app_config=None, reasoning_instruction=None):
    """
    Build a complete prompt string using the prompt configuration,
    retrieved documents, user query, and optional reasoning instructions.

    Parameters
    ----------
    config : dict
        Prompt configuration dictionary (e.g., prompt_config["rag_assistant_prompt"]).
    documents : list[str]
        List of retrieved and formatted reviews.
    query : str
        The user's input question.
    app_config : dict, optional
        Full application config dictionary (not required if reasoning_instruction provided).
    reasoning_instruction : str, optional
        Reasoning strategy instruction to inject into the prompt (e.g., Chain-of-Thought).

    Returns
    -------
    str
        Fully formatted prompt to send to the language model.
    """
    context = "\n".join(documents)
    sections = []

    # Prompt structure
    sections.append(format_prompt_section("Role:", config.get("role", "")))
    sections.append(format_prompt_section("Style / Tone:", config.get("style_or_tone", [])))
    sections.append(format_prompt_section("Instruction:", config.get("instruction", "")))
    sections.append(format_prompt_section("Output Constraints:", config.get("output_constraints", [])))
    sections.append(format_prompt_section("Output Format:", config.get("output_format", [])))

    # Add reasoning strategy if provided explicitly
    if reasoning_instruction:
        sections.append(format_prompt_section("Reasoning Strategy:", reasoning_instruction))
    # Otherwise, fallback to app_config for CoT if available
    elif app_config and "reasoning_strategies" in app_config and "CoT" in app_config["reasoning_strategies"]:
        cot_instruction = app_config["reasoning_strategies"]["CoT"]
        sections.append(format_prompt_section("Reasoning Strategy:", cot_instruction))        

    # Add context and query
    sections.append("Context:\n" + context)
    sections.append("User's question:\n" + query)

    return "\n\n".join(sections)