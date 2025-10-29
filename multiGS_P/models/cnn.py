# multiGS_P/models/cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from .base import BaseModel


class CNN(BaseModel):
    """PyTorch-based 1D CNN model for genomic prediction in multiGS_P."""

    def __init__(self, input_dim=None, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_channels = kwargs.get("conv_channels", [32, 64])
        self.kernel_sizes = kwargs.get("kernel_sizes", [5, 3])
        self.hidden_layers = kwargs.get("hidden_layers", [128])
        self.activation = kwargs.get("activation", "relu")
        self.dropout = kwargs.get("dropout", 0.0)
        self.lr = kwargs.get("lr", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.epochs = kwargs.get("epochs", 50)
        self.batch_size = kwargs.get("batch_size", 32)
        self.device = kwargs.get("device", "cpu")

        # Training control
        self.patience = kwargs.get("patience", 10)
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.grad_clip = kwargs.get("grad_clip", None)
        self.early_stopping = kwargs.get("early_stopping", True)

        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _build_model(self, input_dim):
        """Build CNN layers with conv, dropout, and FC."""
        act_fn = self._get_activation()

        # Convolutional layers
        conv_layers = []
        in_channels = 1
        for out_channels, k in zip(self.conv_channels, self.kernel_sizes):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k))
            conv_layers.append(act_fn)
            if self.dropout > 0:
                conv_layers.append(nn.Dropout(self.dropout))
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)

        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            conv_out = self.conv(dummy)
            fc_input_dim = conv_out.view(1, -1).shape[1]

        # Fully connected layers
        fc_layers = []
        in_dim = fc_input_dim
        for h in self.hidden_layers:
            fc_layers.append(nn.Linear(in_dim, h))
            fc_layers.append(act_fn)
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout))
            in_dim = h
        fc_layers.append(nn.Linear(in_dim, self.output_dim))
        self.fc = nn.Sequential(*fc_layers)

        # Combine
        self.model = nn.Sequential(self.conv, nn.Flatten(), self.fc).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Scheduler with warmup
        warmup_epochs = max(1, int(self.epochs * self.warmup_ratio))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 1.0
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def fit(self, X, y, X_val=None, y_val=None):
        if self.model is None:
            self._build_model(X.shape[1])

        # Auto 10% val split if none provided
        auto_split = False
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=self.params.get("random_state", 42))
            auto_split = True

        if hasattr(y, "values"): y = y.values
        if hasattr(y_val, "values"): y_val = y_val.values

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
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
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.history["train_loss"].append(avg_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(self.device)
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

            self.scheduler.step()

        # Retrain full dataset if auto-split
        if auto_split:
            print(f"Retraining on full dataset for {best_epoch} epochs...")

            self._build_model(X.shape[1])  # rebuild fresh

            X_full = np.asarray(np.concatenate([X, X_val]), dtype=np.float32)
            y_full = np.asarray(np.concatenate([y, y_val]), dtype=np.float32)

            X_tensor = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1).to(self.device)
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
            if best_state is not None:
                self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        if self.model is None:
            self._build_model(X.shape[1])
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().reshape(-1)
        return preds

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized model.")
        torch.save({
            "state_dict": self.model.state_dict(),
            "conv_channels": self.conv_channels,
            "kernel_sizes": self.kernel_sizes,
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
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }, path)
        return self

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.conv_channels = checkpoint.get("conv_channels", [32, 64])
        self.kernel_sizes = checkpoint.get("kernel_sizes", [5, 3])
        self.hidden_layers = checkpoint.get("hidden_layers", [128])
        self.activation = checkpoint.get("activation", "relu")
        self.dropout = checkpoint.get("dropout", 0.0)
        self.lr = checkpoint.get("lr", 0.001)
        self.weight_decay = checkpoint.get("weight_decay", 1e-4)
        self.epochs = checkpoint.get("epochs", 50)
        self.batch_size = checkpoint.get("batch_size", 32)
        self.patience = checkpoint.get("patience", 10)
        self.warmup_ratio = checkpoint.get("warmup_ratio", 0.1)
        self.grad_clip = checkpoint.get("grad_clip", None)
        self.input_dim = checkpoint.get("input_dim")
        self.output_dim = checkpoint.get("output_dim", 1)

        self._build_model(self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        return self
