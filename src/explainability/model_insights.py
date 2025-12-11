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
        }
        return explanation

    def globalImportance(self) -> Dict[str, float]:
        if hasattr(self.explainer, "expected_value") and isinstance(
            self.explainer, shap.TreeExplainer
        ):
            sampleShapVals = self.explainer.shap_values(self.refData)
            if isinstance(sampleShapVals, list):
                sampleShapVals = sampleShapVals[1]
            meanAbsShap = np.mean(np.abs(sampleShapVals), axis=0)
            globalImp = dict(zip(self.featNames, meanAbsShap))
            return {
                k: float(v)
                for k, v in sorted(
                    globalImp.items(), key=lambda item: item[1], reverse=True
                )
            }
        else:
            return {}


if __name__ == "__main__":
    import lightgbm as lgb
    from sklearn.datasets import make_classification
    import json

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    featureNames = [f"feat_{i}" for i in range(X.shape[1])]
    dfTrain = pd.DataFrame(X, columns=featureNames)
    dfTrain["isTarget"] = y

    lgbmModel = lgb.LGBMClassifier(random_state=42, n_estimators=50)
    lgbmModel.fit(dfTrain[featureNames], dfTrain["isTarget"])

    modelExp = ModelInsights(
        mdl=lgbmModel,
        featNames=featureNames,
        trainData=dfTrain.sample(100, random_state=1),
        expType="tree",
    )

    queryInstanceDf = (
        dfTrain[dfTrain["isTarget"] == 1]
        .sample(1, random_state=1)
        .drop(columns=["isTarget"])
    )
    queryInstance = queryInstanceDf.iloc[0].to_dict()

    shapValues = modelExp.explainPoint(queryInstance)
    print(
        f"Point pred prob (target): {lgbmModel.predict_proba(queryInstanceDf)[0][1]:.4f}"
    )
    print(f"SHAP Values: {json.dumps(shapValues, indent=2)}")

    globalImp = modelExp.globalImportance()
    print(json.dumps(globalImp, indent=2))
