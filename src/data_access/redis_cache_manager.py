import redis
import logging
import json
from typing import Dict, Any, Optional
from datetime import timedelta

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class RedisCacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.redis_host = config.get("redis_host", "localhost")
        self.redis_port = config.get("redis_port", 6379)
        self.redis_db = config.get("redis_db", 0)
        self.redis_client = redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True,
        )
        self.default_ttl_seconds = config.get("default_ttl_seconds", 3600)  # 1 hour
        logger.info(
            f"RedisCacheManager initialized for {self.redis_host}:{self.redis_port}/{self.redis_db}."
        )

    def set_value(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        try:
            serialized_value = json.dumps(value)
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            self.redis_client.setex(key, ttl, serialized_value)
            logger.debug(f"Set key '{key}' in Redis with TTL {ttl}s.")
            return True
        except Exception as e:
            logger.error(f"Error setting key '{key}' in Redis: {e}", exc_info=True)
            return False

    def get_value(self, key: str) -> Optional[Any]:
        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Retrieved key '{key}' from Redis.")
                return json.loads(value)
            logger.debug(f"Key '{key}' not found in Redis.")
            return None
        except Exception as e:
            logger.error(f"Error getting key '{key}' from Redis: {e}", exc_info=True)
            return None

    def delete_key(self, key: str) -> bool:
        try:
            deleted_count = self.redis_client.delete(key)
            if deleted_count > 0:
                logger.debug(f"Deleted key '{key}' from Redis.")
                return True
            logger.debug(f"Key '{key}' not found for deletion in Redis.")
            return False
        except Exception as e:
            logger.error(f"Error deleting key '{key}' from Redis: {e}", exc_info=True)
            return False

    def rpush(self, key: str, value: Any) -> bool:
        try:
            serialized_value = json.dumps(value)
            self.redis_client.rpush(key, serialized_value)
            logger.debug(f"Pushed value to list '{key}' in Redis.")
            return True
        except Exception as e:
            logger.error(f"Error pushing to list '{key}' in Redis: {e}", exc_info=True)
            return False

    def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        try:
            values = self.redis_client.lrange(key, start, end)
            if values:
                logger.debug(f"Retrieved list '{key}' from Redis.")
                return [json.loads(v) for v in values]
            logger.debug(f"List '{key}' not found or empty in Redis.")
            return []
        except Exception as e:
            logger.error(f"Error getting list '{key}' from Redis: {e}", exc_info=True)
            return []

    def check_connection(self) -> bool:
        try:
            self.redis_client.ping()
            logger.info("Successfully pinged Redis.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    print("RedisCacheManager - Module loaded successfully")
    print(
        "Note: Full execution requires Redis running. This module is designed to run within the main fraud detection system"
    )
    exit(0)  # Exit gracefully since this is a library module

    # Dummy config (assuming a local Redis instance)
    redis_config = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "default_ttl_seconds": 10,  # 10 seconds for testing
    }
    cache_manager = RedisCacheManager(redis_config)

    # Test connection
    print(f"\nRedis connection status: {cache_manager.check_connection()}")

    # Test setting and getting
    test_key = "user:123:profile"
    test_value = {"name": "Negin", "rides_completed": 150}
    print(f"\nSetting key '{test_key}'...")
    cache_manager.set_value(test_key, test_value)

    print(f"Getting key '{test_key}'...")
    retrieved_value = cache_manager.get_value(test_key)
    print(f"Retrieved value: {retrieved_value}")

    # Test non-existent key
    print(f"\nGetting non-existent key 'non_existent_key'...")
    non_existent_value = cache_manager.get_value("non_existent_key")
    print(f"Retrieved value: {non_existent_value}")

    # Test TTL (wait for 10 seconds)
    print(
        f"\nWaiting for {redis_config['default_ttl_seconds']} seconds for TTL to expire..."
    )
    import time

    time.sleep(redis_config["default_ttl_seconds"] + 1)
    expired_value = cache_manager.get_value(test_key)
    print(f"Retrieved value after TTL: {expired_value}")

    # Test deletion
    new_key = "temp_key"
    cache_manager.set_value(new_key, "temporary_data")
    print(f"\nDeleting key '{new_key}'...")
    cache_manager.delete_key(new_key)
    print(f"Getting key '{new_key}' after deletion: {cache_manager.get_value(new_key)}")
