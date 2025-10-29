# multiGS_P/models/krr_linear.py

from sklearn.kernel_ridge import KernelRidge
import joblib
from .base import BaseModel


class KRR_LINEAR(BaseModel):
    """Kernel Ridge Regression with Linear kernel wrapper for multiGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = KernelRidge(kernel="linear", **kwargs)

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
        return self