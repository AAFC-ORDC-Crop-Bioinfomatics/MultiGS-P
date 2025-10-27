# omniGS_P/models/mlp_sklearn.py

from sklearn.neural_network import MLPRegressor
import joblib
from .base import BaseModel


class MLP_SKLEARN(BaseModel):
    """
    scikit-learn MLPRegressor wrapped for omniGS_P.
    """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.model = MLPRegressor(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        
        self.model.fit(X, y)

        # Store history if available
        self.history = {
            "loss_curve_": getattr(self.model, "loss_curve_", None),
            "validation_scores_": getattr(self.model, "validation_scores_", None),
            "best_validation_score_": getattr(self.model, "best_validation_score_", None),
            "n_iter_": getattr(self.model, "n_iter_", None),
        }
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)
        return self

    def load(self, path: str):
        self.model = joblib.load(path)
        return self
