
from __future__ import annotations
import joblib
from typing import Any

class ModelLoader:
    """Lightweight wrapper to load and hold the sklearn Pipeline."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, texts: list[str]) -> list[int]:
        return self.model.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> list[float]:
        # If the model exposes predict_proba, return spam probability (class 1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(texts)
            # assume positive class is 1 in column 1
            if probs.shape[1] == 2:
                return probs[:, 1].tolist()
            # fallback: last column
            return probs[:, -1].tolist()
        return [None] * len(texts)
