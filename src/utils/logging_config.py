"""
Centralized Logging Configuration for Fraud Detection System

This module provides standardized logging setup across all system components.
Ensures consistent log levels, formats, and output destinations for debugging and monitoring.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            "datefmt": "%Y-%m-%dT%H:%M:%SZ",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/fraud_detection.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/fraud_detection_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
        },
    },
    "loggers": {
        "fraud_detection": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        },
        "src": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
}


class FraudDetectionLogger:
    """Centralized logger factory for the fraud detection system."""

    _instance: Optional["FraudDetectionLogger"] = None
    _configured: bool = False

    def __new__(cls) -> "FraudDetectionLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._configured:
            self.setup_logging()

    def setup_logging(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
        log_to_file: bool = True,
    ) -> None:
        """
        Configure logging for the entire application.

        Args:
            config: Custom logging configuration dictionary
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to enable file logging
        """
        if config is None:
            config = DEFAULT_LOGGING_CONFIG.copy()

        # Set log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        config["handlers"]["console"]["level"] = log_level.upper()

        # Optionally disable file logging
        if not log_to_file:
            for logger_config in config["loggers"].values():
                if "handlers" in logger_config:
                    logger_config["handlers"] = [
                        h for h in logger_config["handlers"] if not h.endswith("_file")
                    ]

        # Ensure log directory exists
        if log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

        # Apply configuration
        logging.config.dictConfig(config)
        self._configured = True

        # Log the setup
        logger = logging.getLogger("fraud_detection")
        logger.info(
            f"Logging configured with level {log_level}, file logging: {log_to_file}"
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name (typically __name__ from calling module)

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)

    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """
        Dynamically change log level for a specific logger or all loggers.

        Args:
            level: New log level
            logger_name: Specific logger name, or None for root logger
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
        else:
            logging.getLogger().setLevel(numeric_level)


# Global logger instance
logger = FraudDetectionLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.get_logger(name)


def setup_system_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Setup logging for the entire system. Call this once at application startup.

    Args:
        log_level: Default log level
        log_to_file: Enable file logging
    """
    logger.setup_logging(log_level=log_level, log_to_file=log_to_file)


# Example usage in modules:
# from src.utils.logging_config import get_logger
# logger = get_logger(__name__)
# logger.info("Module initialized")
# logger.debug("Detailed debug information")
# logger.error("Error occurred", exc_info=True)
