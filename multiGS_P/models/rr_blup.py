# multiGS_P/models/rr_blup.py

from sklearn.linear_model import Ridge
import joblib
from .base import BaseModel


class RR_BLUP(BaseModel):
    """RR-BLUP model wrapper for multiGS_P using Ridge regression."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = Ridge(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        
        self.model.fit(X, y)
        return self

    def predict(self, X):
        
        return self.model.predict(X)

    def save(self, path: str):
        
        joblib.dump(self.model, path)
        return self

    def load(self, path: str):
        
        self.model = joblib.load(path)