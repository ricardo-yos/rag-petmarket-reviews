import logging
import os
from config.paths import LOGS_DIR


def setup_logger(name: str, level: str = "INFO", log_filename: str = "rag_reviews.log") -> logging.Logger:
    """
    Set up and return a logger that writes logs to both the console and a file.

    This logger is useful for tracking the execution of scripts and modules
    by storing timestamped messages in a specified log file and also printing
    them to the terminal for real-time visibility.

    Parameters
    ----------
    name : str
        Unique name for the logger instance. It is used internally by the logging module
        to manage different loggers.
    level : str, optional
        Logging level to use. Should be one of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        Defaults to 'INFO'.
    log_filename : str, optional
        Name of the file (inside the logs directory) where logs will be saved.
        Defaults to 'rag_reviews.log'.

    Returns
    -------
    logging.Logger
        A configured logger instance with both file and console handlers.

    Notes
    -----
    - If the logger is created more than once with the same name, this function will not
      add duplicate handlers thanks to the hasHandlers() check.
    - Log files are saved in the directory specified by `LOGS_DIR`, defined in `config.paths`.
    - The log format includes timestamp, log level, and the log message.
    """

    # Create or retrieve a logger with the specified name
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # Convert string log level (e.g., "INFO") to the corresponding constant (e.g., logging.INFO)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Ensure that the log directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Full path to the log file
    log_path = os.path.join(LOGS_DIR, log_filename)

    # Define a log message format: timestamp, log level, and the message
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Create a file handler to write logs to the log file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler to also print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Return the fully configured logger
    return logger