# omniGS/models/gblup.py

import numpy as np
from sklearn.linear_model import Ridge
import joblib
from .base import BaseModel


class GBLUP(BaseModel):
    """GBLUP model wrapper for OmniGS."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = Ridge(**kwargs)
        self.X_train = None  # Keep training data for kernel prediction

    def _compute_grm(self, X):
        """Compute the genomic relationship matrix (GRM)."""
        n_markers = X.shape[1]
        return (X @ X.T) / n_markers

    def fit(self, X, y, X_val=None, y_val=None):
        
        self.X_train = X
        K = self._compute_grm(X)
        self.model.fit(K, y)
        return self

    def predict(self, X):
        
        if self.X_train is None:
            raise ValueError("Model must be fit before prediction.")

        
        n_markers = self.X_train.shape[1]
        K_test = (X @ self.X_train.T) / n_markers
        return self.model.predict(K_test)

    def save(self, path: str):
        
        joblib.dump((self.model, self.X_train), path)
        return self

    def load(self, path: str):
        
        self.model, self.X_train = joblib.load(path)
        return self