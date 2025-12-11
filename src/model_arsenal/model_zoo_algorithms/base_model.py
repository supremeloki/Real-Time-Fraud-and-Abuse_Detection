from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def get_feature_names(self) -> list:
        pass
