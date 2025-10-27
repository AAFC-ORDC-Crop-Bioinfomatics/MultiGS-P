# omniGS_P/models/stacking.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from .base import BaseModel


class STACKING(BaseModel):
    """
    omniGS_P Stacking Ensemble.
    """

    def __init__(self, base_models=None, meta_model=None, cv=5, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_names = base_models or []
        self.meta_model_name = meta_model or "Ridge"
        self.cv = int(cv)
        self.logger = logger
        self.model_params = kwargs.get("model_params", {})

        # Fitted models and scalers
        self.base_models = []
        self.meta_models = []  # One per trait
        self.meta_scalers = []  # For meta-feature normalization


    def fit(self, X, y, X_val=None, y_val=None):

        from .registry import get_model_class

        # ------------------------------
        # Prepare target
        # ------------------------------
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_targets = y.shape
        n_bases = len(self.base_model_names)
        if self.logger:
            self.logger.info(f"Training with {n_samples} samples, {n_targets} traits, {n_bases} base models.")

        # ------------------------------
        # 1. Generate OOF base predictions
        # ------------------------------
        oof_preds = np.zeros((n_samples, n_bases, n_targets), dtype=np.float32)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            if self.logger:
                self.logger.info(f"Fold {fold}/{self.cv} - generating base model predictions...")

            X_train, X_val_fold = X[train_idx], X[val_idx]
            y_train = y[train_idx, :]

            for j, model_name in enumerate(self.base_model_names):
                model_cls = get_model_class(model_name)
                model_params = self.model_params.get(model_name, {})

                # Instantiate and train
                model = model_cls(**model_params)
                model.fit(X_train, y_train)

                # Predict on validation fold
                preds = model.predict(X_val_fold)
                preds = np.asarray(preds)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                if preds.shape[1] != n_targets:
                    preds = np.tile(preds, (1, n_targets))
                oof_preds[val_idx, j, :] = preds

        # ------------------------------
        # 2. Train one meta-model per trait
        # ------------------------------
        self.meta_models = []
        self.meta_scalers = []

        for t in range(n_targets):
            meta_X = oof_preds[:, :, t]  # (n_samples, n_bases)
            meta_y = y[:, t]

            scaler = StandardScaler()
            meta_X_scaled = scaler.fit_transform(meta_X)
            self.meta_scalers.append(scaler)

            meta_cls = get_model_class(self.meta_model_name)
            meta_params = self.model_params.get(self.meta_model_name, {})
            # default to Ridge for stability
            if self.meta_model_name.lower() in ("linear", "ridge"):
                meta_params.setdefault("alpha", 1.0)
                meta = Ridge(**meta_params)
            else:
                meta = meta_cls(**meta_params)

            meta.fit(meta_X_scaled, meta_y)
            self.meta_models.append(meta)

            if self.logger:
                self.logger.info(
                    f"Trained meta-model for trait {t+1}/{n_targets} "
                    f"({self.meta_model_name}) on OOF features ({self.cv}-fold)."
                )

        # ------------------------------
        # 3. Retrain all base models on full data
        # ------------------------------
        self.base_models = []
        for model_name in self.base_model_names:
            model_cls = get_model_class(model_name)
            model_params = self.model_params.get(model_name, {})
            model = model_cls(**model_params)
            model.fit(X, y)
            self.base_models.append(model)

            if self.logger:
                self.logger.info(f"Base model {model_name} retrained on full dataset.")

        if self.logger:
            self.logger.info(f"Completed ensemble training with {len(self.meta_models)} meta-models.")

        return self


    def predict(self, X):
        """Predict using base model ensemble + meta models."""
        if not self.base_models or not self.meta_models:
            raise RuntimeError("STACKING model not trained or loaded.")

        n_targets = len(self.meta_models)
        n_bases = len(self.base_models)
        n_samples = X.shape[0]

        # Collect base model predictions
        base_preds = np.zeros((n_samples, n_bases, n_targets), dtype=np.float32)
        for j, model in enumerate(self.base_models):
            preds = model.predict(X)
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if preds.shape[1] != n_targets:
                preds = np.tile(preds, (1, n_targets))
            base_preds[:, j, :] = preds

        # Combine base predictions with each meta-model
        final_preds = np.zeros((n_samples, n_targets), dtype=np.float32)
        for t in range(n_targets):
            meta_X = base_preds[:, :, t]
            meta_X_scaled = self.meta_scalers[t].transform(meta_X)
            meta_model = self.meta_models[t]
            final_preds[:, t] = meta_model.predict(meta_X_scaled).reshape(-1)

        if self.logger:
            self.logger.info(f"Generated final predictions for {n_samples} samples.")

        return final_preds if n_targets > 1 else final_preds.ravel()


    def save(self, path: str):
        """Save base models, meta models, and scalers."""
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(self.base_models):
            name = self.base_model_names[i]
            model.save(os.path.join(path, f"base_{name}.pkl"))

        for i, meta_model in enumerate(self.meta_models):
            joblib.dump(meta_model, os.path.join(path, f"meta_trait{i+1}_{self.meta_model_name}.pkl"))
            joblib.dump(self.meta_scalers[i], os.path.join(path, f"scaler_trait{i+1}.pkl"))

        if self.logger:
            self.logger.info(f"Saved ensemble ({len(self.meta_models)} traits) to {path}")
        return self

    def load(self, path: str):
        """Load base models, meta models, and scalers."""
        from .registry import get_model_class
        self.base_models = []
        for name in self.base_model_names:
            cls = get_model_class(name)
            model = cls()
            model.load(os.path.join(path, f"base_{name}.pkl"))
            self.base_models.append(model)

        self.meta_models = []
        self.meta_scalers = []
        trait_idx = 1
        while True:
            meta_path = os.path.join(path, f"meta_trait{trait_idx}_{self.meta_model_name}.pkl")
            scaler_path = os.path.join(path, f"scaler_trait{trait_idx}.pkl")
            if not os.path.exists(meta_path):
                break
            self.meta_models.append(joblib.load(meta_path))
            self.meta_scalers.append(joblib.load(scaler_path))
            trait_idx += 1

        if self.logger:
            self.logger.info(f"Loaded ensemble with {len(self.meta_models)} traits from {path}")
        return self