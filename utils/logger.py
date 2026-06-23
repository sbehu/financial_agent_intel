import logging
import sys

def setup_custom_logger(name):
    """
    Produces a unified, high-visibility stream logger for tracking 
    agent routing decisions and RAG data pipeline operations.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Direct logs cleanly to standard output streams for Docker visibility
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate log entries across modules
    if not logger.handlers:
        logger.addHandler(screen_handler)
        
    return logger