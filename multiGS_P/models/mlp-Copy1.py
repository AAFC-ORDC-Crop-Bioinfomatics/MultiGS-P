# omniGS/models/mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from .base import BaseModel


class MLP(BaseModel):
    """PyTorch-based Multi-Layer Perceptron for OmniGS."""

    def __init__(self, input_dim=None, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = kwargs.get("hidden_layers", [128, 64])
        self.activation = kwargs.get("activation", "relu")
        self.dropout = kwargs.get("dropout", 0.0)
        self.lr = kwargs.get("lr", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.epochs = kwargs.get("epochs", 50)
        self.batch_size = kwargs.get("batch_size", 32)
        self.device = kwargs.get("device", "cpu")

        # Training control
        self.patience = kwargs.get("patience", 10)        # early stopping patience
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)  # fraction of epochs for LR warmup
        self.grad_clip = kwargs.get("grad_clip", None)    
        self.early_stopping = kwargs.get("early_stopping", True)

        # Model will be built later
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _build_model(self, input_dim):
        """Build network based on hidden layers + dropout."""
        self.input_dim = input_dim
        layers = []
        in_dim = input_dim
        act_fn = self._get_activation()
        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Scheduler with warmup (linear warmup then constant LR)
        warmup_epochs = max(1, int(self.epochs * self.warmup_ratio))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 1.0
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def fit(self, X, y, X_val=None, y_val=None):
        if self.model is None:
            self._build_model(X.shape[1])

        # -------------------------
        # Early stopping search
        # -------------------------
        auto_split = False
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.1, random_state=self.params.get("random_state", 42)
            )
            auto_split = True  

        # Convert to tensors
        y = np.asarray(y, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = self.epochs
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()

                # Gradient clipping
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.history["train_loss"].append(avg_loss)

            # Validation loss
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device)
                val_preds = self.model(X_val_tensor)
                val_loss = self.criterion(val_preds, y_val_tensor).item()
            self.history["val_loss"].append(val_loss)

            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}, best epoch = {best_epoch}")
                        break

            # Scheduler step
            self.scheduler.step()

        
        if auto_split:
            print(f"Retraining on full dataset for {best_epoch} epochs...")

            # Rebuild model fresh
            self._build_model(X.shape[1])

            X_full = np.asarray(np.concatenate([X, X_val]), dtype=np.float32)
            y_full = np.asarray(np.concatenate([y, y_val]), dtype=np.float32)

            X_tensor = torch.tensor(X_full, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_full, dtype=torch.float32).reshape(-1, 1).to(self.device)

            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(best_epoch):
                self.model.train()
                for xb, yb in loader:
                    self.optimizer.zero_grad()
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

            print("Retraining complete.")

        else:
            # If CV mode, just restore best weights
            if best_state is not None:
                self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been built or trained yet.")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().reshape(-1)
        return preds

    def save(self, path: str):
        """Save trained PyTorch model and architecture settings."""
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized model. Train or build it first.")
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "warmup_ratio": self.warmup_ratio,
            "grad_clip": self.grad_clip,
        }, path)
        return self

    def load(self, path: str):
        """Load trained PyTorch model and rebuild architecture."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint.get("input_dim")
        self.output_dim = checkpoint.get("output_dim", 1)
        self.hidden_layers = checkpoint.get("hidden_layers", [128, 64])
        self.activation = checkpoint.get("activation", "relu")
        self.dropout = checkpoint.get("dropout", 0.0)
        self.lr = checkpoint.get("lr", 0.001)
        self.weight_decay = checkpoint.get("weight_decay", 1e-4)
        self.epochs = checkpoint.get("epochs", 50)
        self.batch_size = checkpoint.get("batch_size", 32)
        self.patience = checkpoint.get("patience", 10)
        self.warmup_ratio = checkpoint.get("warmup_ratio", 0.1)
        self.grad_clip = checkpoint.get("grad_clip", None)

        # Rebuild model before loading weights
        self._build_model(self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        return self