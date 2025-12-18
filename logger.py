import logging
import os


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    
    logger = logging.getLogger("agent")
    
    return logger

