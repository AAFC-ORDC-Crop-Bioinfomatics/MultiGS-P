# omniGS/models/svr_poly.py

from sklearn.svm import SVR
import joblib
from .base import BaseModel


class SVR_POLY(BaseModel):
    """Support Vector Regression with Polynomial kernel wrapper for OmniGS."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = SVR(kernel="poly", **kwargs)

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