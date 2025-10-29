# src/data_access/feature_store_client.py

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager
from src.data_access.data_lake_client import DataLakeClient  # Assuming DataLakeClient
from src.feature_forge.realtime_feature_engineer import (
    RealtimeFeatureEngineer,
)  # For on-the-fly feature calculation
from src.graph_processor.node_embedding_updater import (
    NodeEmbeddingUpdater,
)  # For GNN embeddings

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """
    A unified client for accessing and writing various types of features (batch, real-time, graph).
    Acts as an abstraction layer over different storage technologies (Redis, S3).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = RedisCacheManager(config["redis_config"])
        self.data_lake_client = DataLakeClient(config["data_lake_config"])
        self.realtime_engineer = RealtimeFeatureEngineer(
            config.get("realtime_feature_windows", [5, 30, 60])
        )
        self.node_embedding_updater = NodeEmbeddingUpdater(
            embedding_dimension=config.get("gnn_embedding_dimension", 64),
            decay_factor=config.get("gnn_embedding_decay_factor", 0.9),
        )
        logger.info("FeatureStoreClient initialized.")

    # --- Real-time Feature Management (via RedisCacheManager) ---

    def _get_realtime_entity_key(self, entity_type: str, entity_id: str) -> str:
        return f"rt_features:{entity_type}:{entity_id}"

    def get_realtime_features(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Fetches real-time aggregated features for an entity from Redis."""
        features = self.redis_client.get_value(
            self._get_realtime_entity_key(entity_type, entity_id)
        )
        return features if features else {}

    def update_and_get_realtime_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an incoming event, updates real-time feature history,
        and generates current real-time features for the event's entities.
        """
        # The RealtimeFeatureEngineer manages its own internal history (deque)
        # and generates features based on that.
        # It's stateful within the InferenceEngine/FeatureStoreClient instance.
        features = self.realtime_engineer.generate_features(event)

        # Optionally, persist these generated features to Redis for direct lookup later,
        # or rely solely on `realtime_engineer`'s internal state.
        # For simplicity in this abstract client, we'll just return them,
        # and assume the engineer's internal state is sufficient for subsequent calls
        # within the current process lifetime.

        # If persisted, it would look like:
        # if event.get("user_id"):
        #     user_rt_features = {k:v for k,v in features.items() if k.startswith('user_')}
        #     self.redis_client.set_value(self._get_realtime_entity_key("user", event["user_id"]), user_rt_features)
        # if event.get("driver_id"):
        #     driver_rt_features = {k:v for k,v in features.items() if k.startswith('driver_')}
        #     self.redis_client.set_value(self._get_realtime_entity_key("driver", event["driver_id"]), driver_rt_features)

        return features

    # --- Batch Feature Management (via DataLakeClient/pre-loaded) ---

    def _get_batch_entity_key(self, entity_type: str, entity_id: str) -> str:
        # Batch features might be too large for Redis, often fetched from a persistent store (S3, Cassandra)
        # or a dedicated batch feature store, possibly cached locally.
        return f"batch_features:{entity_type}:{entity_id}"  # For potential local cache/Redis copy

    def get_batch_features(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Fetches batch-computed features for an entity."""
        # In a real system, this would query a dedicated batch feature store
        # For this demo, we can simulate loading from pre-computed CSVs in data_vault
        if entity_type == "user":
            batch_df = self.data_lake_client.load_dataframe_from_s3(
                s3_prefix="batch_features",
                s3_filename="batch_user_features.csv",
                file_format="csv",
            )
            if batch_df is not None:
                user_features = batch_df[batch_df["user_id"] == entity_id].to_dict(
                    orient="records"
                )
                return user_features[0] if user_features else {}
        elif entity_type == "driver":
            batch_df = self.data_lake_client.load_dataframe_from_s3(
                s3_prefix="batch_features",
                s3_filename="batch_driver_features.csv",
                file_format="csv",
            )
            if batch_df is not None:
                driver_features = batch_df[batch_df["driver_id"] == entity_id].to_dict(
                    orient="records"
                )
                return driver_features[0] if driver_features else {}
        return {}

    # --- Graph Features (from DataLakeClient/embeddings) ---

    def get_graph_static_features(self, node_id: str) -> Dict[str, Any]:
        """Fetches static graph features (e.g., centrality) for a node."""
        graph_features_df = self.data_lake_client.load_dataframe_from_s3(
            s3_prefix="graph_topology_data",
            s3_filename="processed_graph_features.csv",
            file_format="csv",
        )
        if graph_features_df is not None:
            node_features = graph_features_df[
                graph_features_df["node_id"] == node_id
            ].to_dict(orient="records")
            return node_features[0] if node_features else {}
        return {}

    def update_and_get_node_embeddings(
        self, event: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Updates node embeddings based on the current event and returns the latest embeddings.
        """
        self.node_embedding_updater.process_event(event)
        return self.node_embedding_updater.get_combined_embeddings(
            event.get("user_id"), event.get("driver_id")
        )

    # --- Unified Feature Retrieval for an Event ---

    def get_features_for_event(
        self, event: Dict[str, Any], current_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Aggregates all relevant features for a given event, including real-time, batch, and graph.
        """
        all_features = {}

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        # 1. Base event features (directly from the event)
        all_features.update(
            {
                "fare_amount": event.get("fare_amount", 0.0),
                "distance_km": event.get("distance_km", 0.0),
                "duration_min": event.get("duration_min", 0.0),
                "promo_code_used": 1.0 if event.get("promo_code_used") else 0.0,
                # Add other direct event features as needed
            }
        )

        # 2. Real-time engineered features (dynamic, updated per event)
        rt_features = self.update_and_get_realtime_features(event)
        all_features.update(rt_features)

        # 3. Batch features (static/slowly changing, for user/driver)
        if user_id:
            user_batch_feats = self.get_batch_features("user", user_id)
            all_features.update(
                {
                    f"user_batch_{k}": v
                    for k, v in user_batch_feats.items()
                    if k != "user_id"
                }
            )
        if driver_id:
            driver_batch_feats = self.get_batch_features("driver", driver_id)
            all_features.update(
                {
                    f"driver_batch_{k}": v
                    for k, v in driver_batch_feats.items()
                    if k != "driver_id"
                }
            )

        # 4. Graph features (static/embeddings)
        if user_id:
            user_graph_static_feats = self.get_graph_static_features(user_id)
            all_features.update(
                {
                    f"user_graph_{k}": v
                    for k, v in user_graph_static_feats.items()
                    if k not in ["node_id", "node_type", "is_collusion_suspect"]
                }
            )
        if driver_id:
            driver_graph_static_feats = self.get_graph_static_features(driver_id)
            all_features.update(
                {
                    f"driver_graph_{k}": v
                    for k, v in driver_graph_static_feats.items()
                    if k not in ["node_id", "node_type", "is_collusion_suspect"]
                }
            )

        # 5. Node embeddings (dynamic, updated with event, for GNN models)
        node_embeddings = self.update_and_get_node_embeddings(event)
        if "user_embedding" in node_embeddings:
            all_features.update(
                {
                    f"user_emb_{i}": val
                    for i, val in enumerate(node_embeddings["user_embedding"])
                }
            )
        if "driver_embedding" in node_embeddings:
            all_features.update(
                {
                    f"driver_emb_{i}": val
                    for i, val in enumerate(node_embeddings["driver_embedding"])
                }
            )

        logger.info(
            f"Aggregated {len(all_features)} features for event {event.get('event_id')}."
        )
        return all_features


if __name__ == "__main__":
    print("FeatureStoreClient - Module loaded successfully")
    print(
        "Note: Full execution requires Redis and S3 access. This module is designed to run within the main fraud detection system"
    )
    exit(0)  # Exit gracefully since this is a library module

    # Mock configuration for demo
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "config"
    data_vault_dir = project_root / "data_vault"

    # Ensure dummy data_vault directories and files exist for demo
    (data_vault_dir / "batch_features").mkdir(parents=True, exist_ok=True)
    (data_vault_dir / "graph_topology_data").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "user_id": ["test_user_1", "test_user_2"],
            "user_lifetime_rides": [100, 50],
            "user_lifetime_avg_fare": [50000, 40000],
        }
    ).to_csv(data_vault_dir / "batch_features" / "batch_user_features.csv", index=False)
    pd.DataFrame(
        {
            "driver_id": ["test_driver_A", "test_driver_B"],
            "driver_lifetime_rides": [200, 120],
            "driver_lifetime_avg_fare": [60000, 55000],
        }
    ).to_csv(
        data_vault_dir / "batch_features" / "batch_driver_features.csv", index=False
    )
    pd.DataFrame(
        {
            "node_id": ["test_user_1", "test_driver_A", "test_user_2"],
            "node_type": ["user", "driver", "user"],
            "degree_centrality": [0.5, 0.7, 0.3],
            "community_id": [1, 1, 2],
        }
    ).to_csv(
        data_vault_dir / "graph_topology_data" / "processed_graph_features.csv",
        index=False,
    )

    # Dummy config
    dummy_config = {
        "redis_config": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "default_ttl_seconds": 3600,
        },
        "data_lake_config": {
            "s3_bucket_name": "dummy-bucket",
            "aws_region": "eu-central-1",
        },
        "realtime_feature_windows": [1, 5],  # 1 and 5 minute windows for demo
        "gnn_embedding_dimension": 4,
    }

    # Mock DataLakeClient to read from local files for demo
    class MockDataLakeClient(DataLakeClient):
        def __init__(self, config_data):
            super().__init__(config_data)
            self.s3_bucket = "dummy-bucket"  # Override to prevent actual S3 calls
            logger.info(
                "MockDataLakeClient initialized, will read from local data_vault."
            )

        def load_dataframe_from_s3(
            self, s3_prefix: str, s3_filename: str, file_format: str = "csv"
        ) -> Optional[pd.DataFrame]:
            full_path = data_vault_dir / s3_prefix / s3_filename
            if full_path.exists():
                logger.info(f"Mock loading from local path: {full_path}")
                if file_format == "csv":
                    return pd.read_csv(full_path)
                elif file_format == "json":
                    return pd.read_json(full_path, lines=True)
            logger.warning(f"Mock file not found: {full_path}")
            return None

        def save_dataframe_to_s3(
            self,
            df: pd.DataFrame,
            s3_prefix: str,
            s3_filename: str,
            file_format: str = "csv",
        ) -> bool:
            logger.info(f"Mock saving DataFrame to S3 (no actual S3 interaction).")
            return True

    fs_client = FeatureStoreClient(dummy_config)
    fs_client.data_lake_client = MockDataLakeClient(
        dummy_config["data_lake_config"]
    )  # Inject mock

    # Test Redis connection
    try:
        fs_client.redis_client.check_connection()
    except Exception as e:
        print(
            f"Could not connect to Redis: {e}. Ensure Redis is running on localhost:6379."
        )
        exit()

    # Simulate an incoming event
    test_event = {
        "event_id": "test_event_001",
        "event_timestamp": datetime.now().isoformat(),
        "event_type": "ride_requested",
        "user_id": "test_user_1",
        "driver_id": "test_driver_A",
        "fare_amount": 70000.0,
        "distance_km": 12.0,
        "duration_min": 25.0,
        "promo_code_used": None,
        "ip_address": "192.168.1.50",
        "device_info": "ios_device_10",
    }

    print("\n--- Getting features for test_event_001 ---")
    features_for_event = fs_client.get_features_for_event(test_event, datetime.now())
    print(json.dumps(features_for_event, indent=2))

    # Simulate another event for the same user/driver to see real-time feature updates
    import time

    time.sleep(1)  # Simulate time passing
    test_event_2 = {
        "event_id": "test_event_002",
        "event_timestamp": datetime.now().isoformat(),
        "event_type": "ride_completed",
        "user_id": "test_user_1",
        "driver_id": "test_driver_A",
        "fare_amount": 65000.0,
        "distance_km": 11.0,
        "duration_min": 22.0,
        "promo_code_used": "PROMO_X",
        "ip_address": "192.168.1.50",
        "device_info": "ios_device_10",
    }
    print(
        "\n--- Getting features for test_event_002 (should show updated real-time and embeddings) ---"
    )
    features_for_event_2 = fs_client.get_features_for_event(
        test_event_2, datetime.now()
    )
    print(json.dumps(features_for_event_2, indent=2))

    # Clean up dummy data_vault files
    (data_vault_dir / "batch_features" / "batch_user_features.csv").unlink(
        missing_ok=True
    )
    (data_vault_dir / "batch_features" / "batch_driver_features.csv").unlink(
        missing_ok=True
    )
    (data_vault_dir / "graph_topology_data" / "processed_graph_features.csv").unlink(
        missing_ok=True
    )
    (data_vault_dir / "batch_features").rmdir()
    (data_vault_dir / "graph_topology_data").rmdir()
    print("\nCleaned up dummy data_vault files.")
