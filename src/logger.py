"""Enhanced logging configuration using loguru.

Provides:
  - Rotating file logs (50 MB, 30-day retention)
  - Structured JSON output option (enabled via ``LOG_JSON=true`` in .env)
  - Log level from ``LOG_LEVEL`` env var
  - Separate error-only log file
  - ``get_recent_logs(n)`` helper for the ``/view_logs`` Telegram command
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

from config import LOG_LEVEL

# ---------------------------------------------------------------------------
# Configuration constants (override via environment variables)
# ---------------------------------------------------------------------------
_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_JSON: bool = os.getenv("LOG_JSON", "false").lower() in ("1", "true", "yes")
_LOG_FILE = _LOG_DIR / "engine_{time}.log"
_ERROR_LOG_FILE = _LOG_DIR / "engine_errors.log"

_CONSOLE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {extra[name]:<24} | {level:<7} | {message}"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {extra[name]:<24} | {level:<7} | {message}"
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _configure() -> None:
    """Configure loguru sinks.  Called once at import time."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()  # remove default handler

    # Console sink
    _loguru_logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=LOG_LEVEL.upper(),
    )

    # Rotating file sink (all levels)
    serialize = _LOG_JSON
    _loguru_logger.add(
        str(_LOG_FILE),
        rotation="50 MB",
        retention="30 days",
        format=_FILE_FORMAT,
        level="DEBUG",
        serialize=serialize,
        enqueue=True,
    )

    # Separate error-only sink
    _loguru_logger.add(
        str(_ERROR_LOG_FILE),
        rotation="20 MB",
        retention="30 days",
        format=_FILE_FORMAT,
        level="ERROR",
        serialize=serialize,
        enqueue=True,
    )


_configure()

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with *name* context."""
    return _loguru_logger.bind(name=name)


def get_recent_logs(n: int = 50) -> str:
    """Return the last *n* lines from the most recent engine log file.

    Used by the ``/view_logs`` Telegram command.  Returns an empty string
    if no log file exists yet.
    """
    log_files = sorted(_LOG_DIR.glob("engine_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        return ""
    try:
        with open(log_files[0], "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""
