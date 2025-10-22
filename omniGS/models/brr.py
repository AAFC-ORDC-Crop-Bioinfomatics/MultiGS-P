# omniGS/models/brr.py

from sklearn.linear_model import BayesianRidge
import joblib
from .base import BaseModel


class BRR(BaseModel):
    """Bayesian Ridge Regression model wrapper for OmniGS."""

    def __init__(self, **kwargs):
        """
        Initialize Bayesian Ridge model with given hyperparameters.
        """
        super().__init__(**kwargs)
        self.model = BayesianRidge(**kwargs)

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
