"""
WAN Logger
==========

Unified logging for all WAN pipelines.
- File + stdout
- Safe for multi-pod
- Timestamped
"""

import logging
import os
from datetime import datetime
from .config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a logger with standard WAN format.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (daily log)
    log_file = os.path.join(
        LOG_DIR,
        f"wan_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
