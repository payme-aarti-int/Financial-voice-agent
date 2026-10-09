"""Central logging configuration.

Call configure_logging() once, as early as possible -- app/__init__.py does
this on import -- so the CLI, the FastAPI app and the standalone scripts
(ingest, query, orpheus_tts self-test, ...) all share one format and one
destination: the console for live feedback, plus a rotating file so nothing
is lost between restarts.

Get a logger anywhere with `logging.getLogger(__name__)`; it inherits this
setup automatically.
"""

from __future__ import annotations

import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEVEL = logging.INFO

# Third-party libraries default to INFO/DEBUG chatter we don't want
# drowning out our own logs.
NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3",
    "mlflow",
    "sentence_transformers", "transformers",
    "chromadb",
    "torch", "kokoro",
)

_configured = False


def configure_logging() -> None:
    """Clean structured logging; idempotent, safe to call from multiple
    entry points (CLI, API, tests)."""
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LEVEL)
    root.handlers = [console_handler, file_handler]

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # torch emits deprecation/UserWarnings on every Kokoro load (weight_norm,
    # torch.jit.script, LSTM dropout) that are not actionable here.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\..*)?")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\..*)?")


# Older entry points import this name.
setup_logging = configure_logging
