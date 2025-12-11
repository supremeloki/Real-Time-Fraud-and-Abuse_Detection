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
            for i, feature in enumerate(self.feature_names)
        }
        logger.debug(f"Generated SHAP explanation for instance.")
        return explanation_dict

    def get_global_feature_importance(self) -> Dict[str, float]:
        # This method is simplified for direct SHAP-based feature importance
        # For TreeExplainer, it often implicitly uses mean(abs(SHAP values))
        # For a full global explanation, you'd calculate SHAP for many instances.
        if hasattr(self.explainer, "expected_value") and isinstance(
            self.explainer, shap.TreeExplainer
        ):
            # TreeExplainer has an explicit feature_importances_ attribute often
            # or you can use .feature_importances_ if the model provides it
            # A common way to get global importance from SHAP is mean absolute SHAP values
            sample_shap_values = self.explainer.shap_values(self.training_data_sample)
            if isinstance(sample_shap_values, list):
                sample_shap_values = sample_shap_values[1]  # For positive class

            mean_abs_shap = np.mean(np.abs(sample_shap_values), axis=0)
            global_importance = dict(zip(self.feature_names, mean_abs_shap))
            return {
                k: float(v)
                for k, v in sorted(
                    global_importance.items(), key=lambda item: item[1], reverse=True
                )
            }
        else:
            logger.warning(
                "Global feature importance not directly supported for this explainer type or model without full dataset."
            )
            return {}


if __name__ == "__main__":
    import lightgbm as lgb
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_train = pd.DataFrame(X, columns=feature_names)
    df_train["is_fraud"] = y

    lgbm_model = lgb.LGBMClassifier(random_state=42, n_estimators=50)
    lgbm_model.fit(df_train[feature_names], df_train["is_fraud"])

    explainer = SHAPExplainer(
        model=lgbm_model,
        feature_names=feature_names,
        training_data_sample=df_train.sample(
            100, random_state=1
        ),  # Small sample for background data
        explainer_type="tree",
    )

    query_instance_df = (
        df_train[df_train["is_fraud"] == 1]
        .sample(1, random_state=1)
        .drop(columns=["is_fraud"])
    )
    query_instance = query_instance_df.iloc[0].to_dict()

    print("--- Explaining a Fraudulent Instance ---")
    shap_values = explainer.explain_instance(query_instance)
    print(
        f"Original instance prediction (fraud probability): {lgbm_model.predict_proba(query_instance_df)[0][1]:.4f}"
    )
    print(
        f"SHAP Values (local feature contributions): {json.dumps(shap_values, indent=2)}"
    )

    print("\n--- Getting Global Feature Importance (Mean Absolute SHAP) ---")
    global_importance = explainer.get_global_feature_importance()
    print(json.dumps(global_importance, indent=2))
