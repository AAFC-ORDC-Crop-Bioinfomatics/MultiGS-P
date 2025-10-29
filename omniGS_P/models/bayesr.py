# multiGS_P/models/bayesr.py

from .base import BaseModel


class BayesR(BaseModel):
    """BayesR model wrapper for multiGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = None  

    def fit(self, X, y, X_val=None, y_val=None):
        
        raise NotImplementedError("BayesR requires external backend (e.g., BGLR in R).")

    def predict(self, X):
        
        raise NotImplementedError("BayesR requires external backend (e.g., BGLR in R).")

    def save(self, path: str):
        
        raise NotImplementedError("Save not implemented for BayesR placeholder.")

    def load(self, path: str):
        
        raise NotImplementedError("Load not implemented for BayesR placeholder.")
