"""Structured logging setup for Kazi."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handler: Optional[logging.Handler] = None,
) -> None:
    """Configure the 'kazi' logger subtree."""
    kazi_logger = logging.getLogger("kazi")
    kazi_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not kazi_logger.handlers:
        h = handler or logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt))
        kazi_logger.addHandler(h)
        kazi_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"kazi.{name}")
