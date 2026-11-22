import logging


def get_logger(name: str, log_file: str, also_console: bool = False) -> logging.Logger:
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if also_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)
