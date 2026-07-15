"""Minimal logging setup without writing into the resource bundle."""

from __future__ import annotations

import logging

from captioner.infrastructure.app_paths import AppPaths


def configure_logging(paths: AppPaths, *, level: int = logging.INFO) -> logging.Logger:
    """Create the log directory and return the application logger."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("captioner")
    logger.setLevel(level)
    return logger
