import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class NodeEmbeddingUpdater:
    def __init__(
        self,
        embedding_dimension: int = 64,
        decay_factor: float = 0.9,
        initial_value: float = 0.1,
    ):
        self.embedding_dimension = embedding_dimension
        self.decay_factor = decay_factor
        self.initial_value = initial_value
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.driver_embeddings: Dict[str, np.ndarray] = {}
        logger.info(
            f"NodeEmbeddingUpdater initialized with dim={embedding_dimension}, decay={decay_factor}."
        )

    def _get_embedding(self, entity_id: str, entity_type: str) -> np.ndarray:
        if entity_type == "user":
            embeddings = self.user_embeddings
        elif entity_type == "driver":
            embeddings = self.driver_embeddings
        else:
            return np.full(self.embedding_dimension, self.initial_value)

        if entity_id not in embeddings:
            embeddings[entity_id] = np.full(
                self.embedding_dimension, self.initial_value
            )
            logger.debug(f"Initialized new {entity_type} embedding for {entity_id}.")
        return embeddings[entity_id]

    def _update_embedding(
        self, entity_id: str, entity_type: str, new_features: Dict[str, Any]
    ):
        current_embedding = self._get_embedding(entity_id, entity_type)

        feature_vector = np.array(
            [
                new_features.get("fare_amount", 0.0),
                new_features.get("distance_km", 0.0),
                new_features.get("duration_min", 0.0),
                1.0 if new_features.get("promo_code_used") else 0.0,
            ]
        )
        if len(feature_vector) < self.embedding_dimension:
            feature_vector = np.pad(
                feature_vector,
                (0, self.embedding_dimension - len(feature_vector)),
                "constant",
            )
        elif len(feature_vector) > self.embedding_dimension:
            feature_vector = feature_vector[: self.embedding_dimension]

        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        updated_embedding = (self.decay_factor * current_embedding) + (
            (1 - self.decay_factor) * feature_vector
        )
        updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)

        if entity_type == "user":
            self.user_embeddings[entity_id] = updated_embedding
        elif entity_type == "driver":
            self.driver_embeddings[entity_id] = updated_embedding
        logger.debug(f"Updated {entity_type} embedding for {entity_id}.")

    def process_event(self, event: Dict[str, Any]):
        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        if user_id:
            self._update_embedding(user_id, "user", event)
        if driver_id:
            self._update_embedding(driver_id, "driver", event)

        logger.info(f"Node embeddings processed for event {event.get('event_id')}.")

    def get_combined_embeddings(
        self, user_id: str = None, driver_id: str = None
    ) -> Dict[str, np.ndarray]:
        combined_embs = {}
        if user_id:
            combined_embs["user_embedding"] = self._get_embedding(user_id, "user")
        if driver_id:
            combined_embs["driver_embedding"] = self._get_embedding(driver_id, "driver")
        return combined_embs
