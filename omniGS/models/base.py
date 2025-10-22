# omniGS/models/base.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all OmniGS models."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        self.history = None  # for DL frameworks if needed

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict outputs for given inputs."""
        pass

    def save(self, path: str):
        """Save trained model (override in subclasses)."""
        raise NotImplementedError(f"{self.name} does not implement save().")

    def load(self, path: str):
        """Load trained model (override in subclasses)."""
        raise NotImplementedError(f"{self.name} does not implement load().")

    # -------------------------
    # Hyperparameter management
    # -------------------------
    @property
    def name(self):
        return self.__class__.__name__

    def get_params(self):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        
    def get_history(self):
        """Return training history (loss curves, metrics) if available."""
        return self.history    