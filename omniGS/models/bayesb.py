# omniGS/models/bayesb.py

from .base import BaseModel


class BayesB(BaseModel):
    """BayesB model wrapper for OmniGS """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = None  # To be linked with R/BGLR backend

    def fit(self, X, y, X_val=None, y_val=None):
        
        raise NotImplementedError("BayesB requires external backend (e.g., BGLR in R).")

    def predict(self, X):
        
        raise NotImplementedError("BayesB requires external backend (e.g., BGLR in R).")

    def save(self, path: str):
        
        raise NotImplementedError("Save not implemented for BayesB placeholder.")

    def load(self, path: str):
        
        raise NotImplementedError("Load not implemented for BayesB placeholder.")
