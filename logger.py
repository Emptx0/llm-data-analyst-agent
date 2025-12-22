import logging
import os


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    
    logger = logging.getLogger("agent")
    
    return logger

