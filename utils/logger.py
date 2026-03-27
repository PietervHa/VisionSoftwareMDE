import logging
from pathlib import Path


def setup_logging():
    """Configure application logging for console and file output."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers when setup is called more than once.
    if getattr(root_logger, "_vision_logging_configured", False):
        return

    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logs_dir / "vision.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Hide per-request Flask development server logs (e.g., GET /result 200).
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    root_logger._vision_logging_configured = True


def get_logger(name):
    return logging.getLogger(name)

