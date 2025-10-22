# omniGS/models/stacking.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .base import BaseModel

class STACKING(BaseModel):
    """Custom Stacking Ensemble for OmniGS ."""

    def __init__(self, base_models=None, meta_model=None, cv=5, logger=None, **kwargs):
        
        super().__init__(**kwargs)
        self.base_model_names = base_models or []
        self.meta_model_name = meta_model
        self.cv = int(cv)
        self.logger = logger

        # All model hyperparameters from config
        self.model_params = kwargs.get("model_params", {})

        
        self.base_models = []
        self.meta_model = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train stacking ensemble.
        - If X_val is provided -> CV : build OOF preds for meta-model.
        - If X_val is None      -> Prediction context: train on full data.
        """
        from .registry import get_model_class
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy().ravel()
        n_samples = X.shape[0]
        n_models = len(self.base_model_names)

        # ---------------------------------------------------
        # CASE 1: CV MODE 
        # ---------------------------------------------------
        if X_val is not None and y_val is not None:
            if self.logger:
                self.logger.info(f"[STACKING] Training in CV mode with {self.cv} folds.")

            oof_preds = np.zeros((n_samples, n_models))
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                if self.logger:
                    self.logger.info(f"[STACKING] Starting fold {fold}/{self.cv}.")

                X_train, y_train = X[train_idx], y[train_idx]
                X_val_fold = X[val_idx]

                for j, model_name in enumerate(self.base_model_names):
                    model_cls = get_model_class(model_name)
                    model = model_cls(**self.model_params.get(model_name, {}))
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val_fold)
                    oof_preds[val_idx, j] = preds

                    if self.logger:
                        self.logger.info(
                            f"[STACKING] Fold {fold}, base model {model_name} trained "
                            f"and predicted {len(val_idx)} samples."
                        )

            # Train meta-model on OOF matrix
            meta_cls = get_model_class(self.meta_model_name)
            self.meta_model = meta_cls(**self.model_params.get(self.meta_model_name, {}))
            self.meta_model.fit(oof_preds, y)

            if self.logger:
                self.logger.info(f"[STACKING] Meta-model {self.meta_model_name} trained on OOF predictions.")

            # Re-train base models on full data 
            self.base_models = []
            for model_name in self.base_model_names:
                model_cls = get_model_class(model_name)
                model = model_cls(**self.model_params.get(model_name, {}))
                model.fit(X, y)
                self.base_models.append(model)
                if self.logger:
                    self.logger.info(f"[STACKING] Base model {model_name} retrained on full data.")

            return self

        # ---------------------------------------------------
        # CASE 2: PREDICTION MODE 
        # ---------------------------------------------------
        else:
            if self.logger:
                self.logger.info(f"[STACKING] Training in Prediction mode (full train set).")

            base_preds = []
            self.base_models = []
            for model_name in self.base_model_names:
                model_cls = get_model_class(model_name)
                model = model_cls(**self.model_params.get(model_name, {}))
                model.fit(X, y)
                self.base_models.append(model)
                preds = model.predict(X).reshape(-1, 1)
                base_preds.append(preds)

                if self.logger:
                    self.logger.info(f"[STACKING] Base model {model_name} trained on full train set.")

            meta_X = np.hstack(base_preds)

            meta_cls = get_model_class(self.meta_model_name)
            self.meta_model = meta_cls(**self.model_params.get(self.meta_model_name, {}))
            self.meta_model.fit(meta_X, y)

            if self.logger:
                self.logger.info(f"[STACKING] Meta-model {self.meta_model_name} trained on stacked predictions.")

            return self

    def predict(self, X):
        """Predict using trained base models + meta model."""
        if self.base_models is None or self.meta_model is None:
            raise RuntimeError("STACKING model is not trained yet.")

        base_preds = []
        for model, model_name in zip(self.base_models, self.base_model_names):
            preds = model.predict(X).reshape(-1, 1)
            base_preds.append(preds)

            if self.logger:
                self.logger.info(f"[STACKING] Base model {model_name} generated predictions for {len(X)} samples.")

        meta_X = np.hstack(base_preds)
        preds = self.meta_model.predict(meta_X)

        if self.logger:
            self.logger.info(f"[STACKING] Meta-model {self.meta_model_name} generated final predictions.")

        return preds

    def save(self, path: str):
        """Save stacking ensemble (base models + meta model)."""
        os.makedirs(path, exist_ok=True)

        for i, model in enumerate(self.base_models):
            model_name = self.base_model_names[i]
            model.save(os.path.join(path, f"base_{model_name}.pkl"))

        self.meta_model.save(os.path.join(path, f"meta_{self.meta_model_name}.pkl"))

        if self.logger:
            self.logger.info(f"[STACKING] Saved ensemble to {path}")

        return self

    def load(self, path: str):
        """Load stacking ensemble (base models + meta model)."""
        from .registry import get_model_class
        self.base_models = []
        for model_name in self.base_model_names:
            cls = get_model_class(model_name)
            model = cls()
            model.load(os.path.join(path, f"base_{model_name}.pkl"))
            self.base_models.append(model)

        meta_cls = get_model_class(self.meta_model_name)
        self.meta_model = meta_cls()
        self.meta_model.load(os.path.join(path, f"meta_{self.meta_model_name}.pkl"))

        if self.logger:
            self.logger.info(f"[STACKING] Loaded ensemble from {path}")

        return self