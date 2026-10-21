"""
Logging in one place, so every message looks the same and is easy to read.

A log line looks like:

    14:32:07 [MIC] Audio level: 0.0412

Never log an API key, password, or any secret here.
"""

import logging
import sys

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Returns a logger that prints to the terminal with a timestamp and a tag."""
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        root = logging.getLogger("robot")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _configured = True
    return logging.getLogger(f"robot.{name}")
