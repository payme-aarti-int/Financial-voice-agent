"""Central logging configuration.

Call setup_logging() once, as early as possible -- app/__init__.py does this
on import -- so the CLI, the FastAPI app and the standalone scripts
(ingest, query, orpheus_tts self-test, ...) all share one format and one
destination: the console for live feedback, plus a rotating file so nothing
is lost between restarts.

Get a logger anywhere with `logging.getLogger(__name__)`; it inherits this
setup automatically.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LEVEL = logging.INFO

_configured = False


def setup_logging() -> None:
    """Idempotent: safe to call from multiple entry points (CLI, API, tests)."""
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LEVEL)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party libraries default to INFO/DEBUG chatter we don't want
    # drowning out our own logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
