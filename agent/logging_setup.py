"""
Central logging for kelvin6k.

Console output stays readable ("[CAMERA] ..." style prefixes preserved via the
logger name), and an optional rotating file log captures everything for
post-mortem analysis of a print run.

Configure with LOG_LEVEL and LOG_FILE in .env.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_configured = False


def configure_logging() -> None:
    """Idempotent: safe to call from any entry point."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "").strip()

    root = logging.getLogger("kelvin6k")
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root.addHandler(console)

    if log_file:
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
            )
            root.addHandler(file_handler)
        except Exception as exc:  # never let logging setup break the app
            root.warning("Could not open log file %s: %s", log_file, exc)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger under the kelvin6k namespace, e.g. get_logger('CAMERA')."""
    configure_logging()
    return logging.getLogger(f"kelvin6k.{name}") if not name.startswith("kelvin6k") else logging.getLogger(name)
