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
