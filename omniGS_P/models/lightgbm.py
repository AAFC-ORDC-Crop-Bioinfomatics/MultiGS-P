# multiGS_P/models/lightgbm.py

from lightgbm import LGBMRegressor
import joblib
from .base import BaseModel
import numpy as np

class LIGHTGBM(BaseModel):
    """LightGBM regression model wrapper for multiGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = LGBMRegressor(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        
        y = np.asarray(y).ravel()  
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val).ravel()
            self.model.fit(X, y, eval_set=[(X_val, y_val)])
        else:
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
