# omniGS/models/bayesa.py
from .base import BaseModel


class BayesA(BaseModel):
    """BayesA model wrapper for OmniGS."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = None  

    def fit(self, X, y, X_val=None, y_val=None):
       
        raise NotImplementedError("BayesA requires external backend (e.g., BGLR in R).")

    def predict(self, X):
      
        raise NotImplementedError("BayesA requires external backend (e.g., BGLR in R).")

    def save(self, path: str):
        
        raise NotImplementedError("Save not implemented for BayesA placeholder.")

    def load(self, path: str):
        
        raise NotImplementedError("Load not implemented for BayesA placeholder.")
