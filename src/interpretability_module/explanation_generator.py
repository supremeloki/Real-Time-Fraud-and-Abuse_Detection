import pandas as pd
import numpy as np
import logging
import argparse
import shap
import json
from pathlib import Path
from typing import Dict, Any, List
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class ExplanationGenerator:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "ExplanationGenerator", self.config["environment"]["log_level"]
        )
        self.feature_names = None  # Will be set by the model
        self.background_data = None  # Will be set by a sample of training data

    def set_model_and_features(
        self, model, feature_names: List[str], background_data: pd.DataFrame = None
    ):
        self.model = model
        self.feature_names = feature_names
        if background_data is not None:
            # Ensure background data has the same columns as feature_names
            self.background_data = background_data[feature_names].sample(
                min(100, len(background_data)), random_state=42
            )
            self.logger.info(
                f"Background data for SHAP set with {len(self.background_data)} samples."
            )
        else:
            self.logger.warning(
                "No background data provided for SHAP. Explanations might be less accurate."
            )

    def generate_shap_explanation(
        self, instance: Dict[str, Any], top_n_features: int = 5
    ) -> Dict[str, float]:
        if (
            self.model is None
            or self.feature_names is None
            or self.background_data is None
        ):
            self.logger.warning(
                "Model, feature names, or background data not set. Cannot generate SHAP explanation."
            )
            return {"error": "Explanation components not initialized."}

        try:
            instance_df = pd.DataFrame([instance])[self.feature_names]

            # Ensure all feature_names are present in instance_df, fill missing with 0
            for col in self.feature_names:
                if col not in instance_df.columns:
                    instance_df[col] = 0.0

            if isinstance(
                self.model,
                (shap.models.LightGBM, shap.models.XGBoost, shap.models.CatBoost),
            ):
                explainer = shap.TreeExplainer(
                    self.model, self.background_data, feature_dependence="independent"
                )
            else:  # For other models, use KernelExplainer, which is model-agnostic but slower
                explainer = shap.KernelExplainer(
                    self.model.predict_proba, self.background_data
                )

            shap_values = explainer.shap_values(instance_df)

            # For multi-class (e.g., GNN output of 2 classes), shap_values will be a list of arrays.
            # For binary classification (like LightGBM for fraud), we take the values for the positive class (index 1).
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values_for_positive_class = shap_values[1][
                    0
                ]  # Assuming first instance, positive class
            else:
                shap_values_for_positive_class = shap_values[
                    0
                ]  # Assuming shap_values is already for a single class or single instance

            feature_importances = dict(
                zip(self.feature_names, shap_values_for_positive_class)
            )
            sorted_features = sorted(
                feature_importances.items(), key=lambda item: abs(item[1]), reverse=True
            )

            top_features_explanation = {
                k: float(v) for k, v in sorted_features[:top_n_features]
            }
            self.logger.debug(
                f"Generated SHAP explanation for instance. Top features: {top_features_explanation}"
            )
            return top_features_explanation
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
            return {"error": str(e)}

    def generate_explanation_for_gnn_node(
        self, node_features: Dict[str, Any], top_n_features: int = 5
    ) -> Dict[str, float]:
        # GNN interpretability is more complex than tabular models.
        # This function provides a simplified "feature importance" based on node features.
        # True GNN explanations involve techniques like GNNExplainer, Grad-CAM for graphs, etc.
        # For simplicity, we'll return a dummy explanation or rely on the LightGBM's feature importance
        # for a "combined" explanation if GNN is an ensemble component.
        self.logger.warning(
            "GNN explanation is highly complex and not fully implemented here. Providing a placeholder."
        )

        # A simple approach could be to find the most impactful features for *this node*
        # based on its feature values, if we assume higher values for certain features
        # correlate with higher fraud scores. This is a heuristic.

        # In a real system, you'd run a dedicated GNN explanation algorithm (e.g., GNNExplainer)
        # on the relevant subgraph.

        # Example: Return highest value features for the node as "important"
        sorted_node_features = sorted(
            node_features.items(), key=lambda item: abs(item[1]), reverse=True
        )
        return {
            k: float(v)
            for k, v in sorted_node_features[:top_n_features]
            if not k.startswith("node_")
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Explanation Generator")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
