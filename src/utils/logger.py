# src/utils/logger.py

import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name (str): Logger name (usually __name__).
        log_file (str, optional): If provided, logs will also be written to this file.
        level (int): Logging level (default: INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
