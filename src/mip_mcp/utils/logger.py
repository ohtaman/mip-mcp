"""Logging utilities."""

import logging
import logging.config

from .config_manager import ConfigManager


def setup_logging(config_manager: ConfigManager | None = None) -> None:
    """Setup logging configuration.

    Args:
        config_manager: Configuration manager instance. If None, creates default.
    """
    if config_manager is None:
        config_manager = ConfigManager()

    logging_config = config_manager.config.logging

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper()),
        format=logging_config.format,
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pyscipopt").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
