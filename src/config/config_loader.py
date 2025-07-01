from pathlib import Path
from typing import Union
import yaml


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """
    Loads a YAML configuration file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If there's an error parsing YAML.
    IOError
        If there's an error reading the file.
    """
    # Ensure file_path is a Path object
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    try:
        # Open and read the file content safely
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)  # Parse YAML into a Python dictionary
    except yaml.YAMLError as e:
        # Raised if YAML parsing fails
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        # Raised if file can't be read
        raise IOError(f"Error reading YAML file: {e}") from e