# omniGS/models/krr_rbf.py

from sklearn.kernel_ridge import KernelRidge
import joblib
from .base import BaseModel


class KRR_RBF(BaseModel):
    """Kernel Ridge Regression with RBF kernel wrapper for OmniGS."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = KernelRidge(kernel="rbf", **kwargs)

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
