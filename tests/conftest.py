"""
Pytest configuration and shared fixtures for the fraud detection system.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logging_config import setup_system_logging


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging for all tests."""
    setup_system_logging(log_level="WARNING", log_to_file=False)


@pytest.fixture
def sample_ride_event():
    """Sample ride event for testing."""
    return {
        "event_id": "test_event_123",
        "user_id": "user_456",
        "ride_id": "ride_789",
        "timestamp": "2023-10-27T12:00:00Z",
        "pickup_location": {"lat": 35.6892, "lon": 139.6917},
        "dropoff_location": {"lat": 35.6580, "lon": 139.7017},
        "fare_amount": 25.50,
        "distance": 5.2,
        "duration": 900,
        "payment_method": "credit_card",
    }


@pytest.fixture
def mock_feature_store():
    """Mock feature store for testing."""

    class MockFeatureStore:
        def __init__(self):
            self.features = {}

        def get_features(self, user_id):
            return self.features.get(user_id, {})

        def store_features(self, user_id, features):
            self.features[user_id] = features

    return MockFeatureStore()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""

    class MockRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

        def delete(self, key):
            if key in self.data:
                del self.data[key]

    return MockRedis()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "model_threshold": 0.8,
        "feature_store_host": "localhost",
        "kafka_bootstrap_servers": "localhost:9092",
        "log_level": "WARNING",
    }
