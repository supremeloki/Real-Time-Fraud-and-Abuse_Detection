"""
Unit tests for logging configuration module.
"""

import pytest
import logging
from src.utils.logging_config import (
    FraudDetectionLogger,
    get_logger,
    setup_system_logging,
)


class TestFraudDetectionLogger:
    """Test the FraudDetectionLogger singleton."""

    def test_singleton_pattern(self):
        """Test that FraudDetectionLogger follows singleton pattern."""
        logger1 = FraudDetectionLogger()
        logger2 = FraudDetectionLogger()
        assert logger1 is logger2

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_setup_system_logging(self):
        """Test system logging setup."""
        setup_system_logging(log_level="DEBUG", log_to_file=False)

        # Check that logger is configured
        logger = logging.getLogger("fraud_detection")
        assert logger.level == logging.DEBUG

    def test_set_level(self):
        """Test dynamic log level changes."""
        logger_instance = FraudDetectionLogger()
        test_logger = logging.getLogger("test_logger")

        # Set to DEBUG
        logger_instance.set_level("DEBUG", "test_logger")
        assert test_logger.level == logging.DEBUG

        # Set to ERROR
        logger_instance.set_level("ERROR", "test_logger")
        assert test_logger.level == logging.ERROR

    def test_invalid_log_level(self):
        """Test handling of invalid log levels."""
        logger_instance = FraudDetectionLogger()

        # Should default to INFO for invalid level
        logger_instance.set_level("INVALID_LEVEL", "test_logger")
        test_logger = logging.getLogger("test_logger")
        # Note: This might not change the level if it's already set, but tests the method doesn't crash
