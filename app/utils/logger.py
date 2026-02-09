import logging
import sys
from functools import lru_cache


def set_logger(log_level: str ='INFO'):

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s]%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt=formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

@lru_cache
def get_logger(name:str)->  logging.Logger:
    return logging.getLogger(name)

class LoggerMixin:
    @property
    def logger(self)->logging.Logger:
        return get_logger(self.__class__.__name__)
    