# multiGS_P/models/svr_rbf.py

from sklearn.svm import SVR
import joblib
from .base import BaseModel


class SVR_RBF(BaseModel):
    """Support Vector Regression with RBF kernel wrapper for multiGS_P"""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = SVR(kernel="rbf", **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        
        self.model.fit(X, y)
        return self

    def predict(self, X):
        
        return self.model.predict(X)

    def save(self, path: str):
        
        joblib.dump(self.model, path)

    def load(self, path: str):
        
        self.model = joblib.load(path)
        return self