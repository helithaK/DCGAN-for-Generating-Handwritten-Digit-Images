import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Sets up a logger for the project.

    Args:
        log_dir (str): Directory to save the log files.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate a log file name with a timestamp
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure the logger
    logger = logging.getLogger("GAN_Logger")
    logger.setLevel(log_level)

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)

    # Stream handler to display logs on the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # Define a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
