"""
app_logger.py — Centralised logging for WakeMate v2.0.
Writes to wakemate.log (rotating, max 5 MB × 3 backups) AND stderr.
Usage:  from app_logger import get_logger
        log = get_logger(__name__)
        log.info("Camera opened")
"""
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wakemate.log")
_loggers: dict[str, logging.Logger] = {}

def get_logger(name: str = "wakemate") -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    if logger.handlers:
        _loggers[name] = logger
        return logger
    logger.setLevel(logging.DEBUG)
    try:
        fh = RotatingFileHandler(
            _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
    except Exception as e:
        print(f"[app_logger WARNING] Could not open log file {_LOG_PATH}: {e}")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt="[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    _loggers[name] = logger
    return logger
