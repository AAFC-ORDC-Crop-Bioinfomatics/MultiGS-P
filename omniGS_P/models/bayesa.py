# omniGS_P/models/bayesa.py
from .base import BaseModel


class BayesA(BaseModel):
    """BayesA model wrapper for omniGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = None  

    def fit(self, X, y, X_val=None, y_val=None):
       
        raise NotImplementedError("")

    def predict(self, X):
      
        raise NotImplementedError("")

    def save(self, path: str):
        
        raise NotImplementedError("")

    def load(self, path: str):
        
        raise NotImplementedError("")
