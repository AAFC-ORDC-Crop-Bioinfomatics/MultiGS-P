# multiGS_P/models/mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .base import BaseModel


class InputDropout(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        return x * mask / (1 - self.p)


class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout=0.5,
                 activation="relu", residual=True, input_dropout=0.05):
        super().__init__()

        # Activation
        act_name = str(activation).lower()
        if act_name == "relu":
            act_fn = nn.ReLU()
        elif act_name == "gelu":
            act_fn = nn.GELU()
        elif act_name == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.residual = residual
        self.input_do = InputDropout(input_dropout)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev = input_dim
        for h in hidden_layers:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h),
                act_fn,
                nn.Dropout(dropout)
            ))
            self.norms.append(nn.LayerNorm(h))
            prev = h

        self.out = nn.Linear(prev, output_dim)

    def forward(self, x):
        x = self.input_do(x)
        h = x
        for block, nm in zip(self.blocks, self.norms):
            z = block(h)
            z = nm(z)
            if self.residual and z.shape == h.shape:
                h = z + h
            else:
                h = z
        return self.out(h)


class MLP(BaseModel):

    def __init__(self, input_dim=None, output_dim=1, **kwargs):
        super().__init__(**kwargs)

        # hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = kwargs.get("hidden_layers", [1024, 512, 256])
        self.activation = kwargs.get("activation", "relu")
        self.dropout = kwargs.get("dropout", 0.5)
        self.lr = kwargs.get("lr", 0.0005)
        self.weight_decay = kwargs.get("weight_decay", 0.0015)
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 32)
        self.device = torch.device(kwargs.get("device", "cpu"))

        # Regularization and training controls
        self.patience = kwargs.get("patience", 10)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.early_stopping = kwargs.get("early_stopping", True)
        self.input_dropout = kwargs.get("input_dropout", 0.05)
        self.residual = True

        # Components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.scaler_y = None  # phenotype normalization

    def _build_model(self, input_dim):
        self.model = MLPNet(
            input_dim=input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            residual=self.residual,
            input_dropout=self.input_dropout
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=8, factor=0.5, verbose=False
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if self.model is None:
            self._build_model(X.shape[1])

        self.scaler_y = StandardScaler()
        y = self.scaler_y.fit_transform(np.asarray(y).reshape(-1, 1))

        # Validation split if not provided
        auto_split = False
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.1, random_state=self.params.get("random_state", 42)
            )
            auto_split = True

        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        
        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).reshape(-1, 1).to(self.device)
        val_X_tensor = torch.from_numpy(X_val).to(self.device)
        val_y_tensor = torch.from_numpy(y_val).reshape(-1, 1).to(self.device)


        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.history = {"train_loss": [], "val_loss": []}

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= len(loader.dataset)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                preds_val = self.model(val_X_tensor)
                val_loss = self.criterion(preds_val, val_y_tensor).item()

            
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.early_stopping:
                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss
                    best_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        
        if auto_split:
            self._retrain_on_full(X, y, X_val, y_val)

        return self


    def _retrain_on_full(self, X, y, X_val, y_val):
        X_full = np.concatenate([X, X_val])
        y_full = np.concatenate([y, y_val])

        X_tensor = torch.tensor(X_full, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_full, dtype=torch.float32).reshape(-1, 1).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(int(self.epochs * 0.5)):  # train half original epochs
            self.model.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
        


    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained or built.")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().reshape(-1, 1)
        if self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds)
        return preds.flatten()


    def save(self, path: str):
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
            "grad_clip": self.grad_clip,
            "scaler_y": self.scaler_y,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.output_dim = checkpoint["output_dim"]
        self.hidden_layers = checkpoint["hidden_layers"]
        self.activation = checkpoint["activation"]
        self.dropout = checkpoint["dropout"]
        self.lr = checkpoint["lr"]
        self.weight_decay = checkpoint["weight_decay"]
        self.epochs = checkpoint["epochs"]
        self.batch_size = checkpoint["batch_size"]
        self.patience = checkpoint["patience"]
        self.grad_clip = checkpoint["grad_clip"]
        self.scaler_y = checkpoint.get("scaler_y", None)

        self._build_model(self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        return self