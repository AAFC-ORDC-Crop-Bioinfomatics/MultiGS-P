# omniGS_P/models/xgboost.py

from xgboost import XGBRegressor
import joblib
from .base import BaseModel


class XGBOOST(BaseModel):
    """XGBoost regression model wrapper for omniGS_P."""

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = XGBRegressor(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
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