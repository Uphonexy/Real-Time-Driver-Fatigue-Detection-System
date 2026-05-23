"""
app_logger.py — Centralised logging for WakeMate v2.0.

Usage:
    from app_logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera opened")
    logger.warning("Audio file missing — using beep fallback")
    logger.error("DB connection failed")

Writes to wakemate.log (rotating, max 5 MB × 3 backups) in the project
root AND to stderr with a concise format.  All existing print() calls in
critical paths should be migrated to this logger.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# ── Log file sits next to the source files ──────────────────────────────────
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wakemate.log")

# Module-level cache: name → Logger, so handlers are never added twice
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "wakemate") -> logging.Logger:
    """
    Return a named logger configured with a RotatingFileHandler and a
    StreamHandler.  Subsequent calls with the same name return the cached
    instance without adding extra handlers.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Avoid double-registering if a parent logger already has handlers
    if logger.handlers:
        _loggers[name] = logger
        return logger

    logger.setLevel(logging.DEBUG)

    # ── File handler (rotating) ──────────────────────────────────────────
    try:
        fh = RotatingFileHandler(
            _LOG_PATH,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)
    except Exception as e:
        # If we cannot open the log file, fall back to stderr only
        print(f"[app_logger WARNING] Could not open log file {_LOG_PATH}: {e}")

    # ── Console handler ──────────────────────────────────────────────────
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(
        logging.Formatter(fmt="[%(levelname)s] %(message)s")
    )
    logger.addHandler(sh)

    # Prevent propagation to the root logger (avoids duplicate messages)
    logger.propagate = False

    _loggers[name] = logger
    return logger
