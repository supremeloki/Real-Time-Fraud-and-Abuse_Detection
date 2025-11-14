# src/explainability/shap_explainer.py

import logging
import pandas as pd
import numpy as np
import shap  # Make sure shap is installed
import json
from typing import Dict, Any, List, Optional, Callable
from src.utils.common_helpers import setup_logging

logger = setup_logging(__name__)


class SHAPExplainer:
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        training_data_sample: pd.DataFrame,
        explainer_type: str = "tree",
    ):
        self.model = model
        self.feature_names = feature_names
        self.training_data_sample = training_data_sample[feature_names]

        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, self.training_data_sample
            )
        else:
            raise ValueError(
                f"Unsupported explainer_type: {explainer_type}. Choose 'tree' or 'kernel'."
            )

        logger.info(f"SHAPExplainer initialized with {explainer_type} explainer.")

    def explain_instance(self, instance: Dict[str, Any]) -> Dict[str, float]:
        query_df = pd.DataFrame([instance])[self.feature_names]

        # Ensure all features are present and in correct order, fill missing with 0 or mean
        for col in self.feature_names:
            if col not in query_df.columns:
                query_df[col] = 0.0  # Or use self.training_data_sample[col].mean()

        query_df = query_df[self.feature_names]  # Reorder columns

        shap_values = self.explainer.shap_values(query_df)

        # For binary classification models (like LightGBM) that predict probabilities
        # shap_values will be a list of two arrays. We usually want the SHAP values
        # for the positive class (class 1, often fraud).
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # SHAP values for class 1 (fraud)

        explanation_dict = {
            feature: float(shap_values[0, i])
