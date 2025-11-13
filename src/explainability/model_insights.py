import logging
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List, Optional, Callable


class ModelInsights:
    def __init__(
        self,
        mdl: Any,
        featNames: List[str],
        trainData: pd.DataFrame,
        expType: str = "tree",
    ):
        self.model = mdl
        self.featNames = featNames
        self.refData = trainData[featNames]
        if expType == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif expType == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, self.refData
            )
        else:
            raise ValueError(f"Unknown explainer type: {expType}.")

    def explainPoint(self, pnt: Dict[str, Any]) -> Dict[str, float]:
        queryDf = pd.DataFrame([pnt])[self.featNames]
        for col in self.featNames:
            if col not in queryDf.columns:
                queryDf[col] = 0.0
        queryDf = queryDf[self.featNames]
        shapVals = self.explainer.shap_values(queryDf)
        if isinstance(shapVals, list):
            shapVals = shapVals[1]
        explanation = {
            feature: float(shapVals[0, i]) for i, feature in enumerate(self.featNames)
