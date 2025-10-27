# omniGS_P/models/svr_linear.py

from sklearn.svm import SVR
import joblib
from .base import BaseModel


class SVR_LINEAR(BaseModel):
    """Support Vector Regression with Linear kernel wrapper for omniGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = SVR(kernel="linear", **kwargs)

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
