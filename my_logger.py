import logging
from uvicorn.logging import DefaultFormatter, AccessFormatter

# ANSI escape codes for colors
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"

class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: COLOR_CYAN,
        logging.INFO: COLOR_GREEN,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_RED,
        logging.CRITICAL: COLOR_RED
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, COLOR_RESET)
        record.levelname = f"{color}{record.levelname}{COLOR_RESET}"
        return super().format(record)

# Setup logger
def setup_logger():

    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()

    # Include timestamp and colored levelname
    formatter = ColorFormatter(
        fmt='%(asctime)s %(levelname)-16s: %(message)s',
        datefmt='%d-%m %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Configure uvicorn loggers
    uvicorn_logger = logging.getLogger("uvicorn")

    # Use your existing ColorFormatter for uvicorn logs too
    uvicorn_logger.handlers.clear()

    # Add your colored formatter to uvicorn
    uvicorn_handler = logging.StreamHandler()
    uvicorn_handler.setFormatter(formatter)  # Your existing ColorFormatter
    uvicorn_logger.addHandler(uvicorn_handler)

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.INFO)
    uvicorn_access.handlers.clear()
    uvicorn_access.addHandler(uvicorn_handler)

    return logger
