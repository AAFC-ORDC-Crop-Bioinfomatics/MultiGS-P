# omniGS/evaluation/metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """Coefficient of Determination (R²)."""
    return r2_score(y_true, y_pred)


def pearson(y_true, y_pred):
    """
    Compute Pearson correlation coefficient 
    """
    try:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("Both input arrays must have the same length.")

        # Subtract means
        y_true_mean = y_true - np.mean(y_true)
        y_pred_mean = y_pred - np.mean(y_pred)

        num = np.sum(y_true_mean * y_pred_mean)
        den = np.sqrt(np.sum(y_true_mean**2)) * np.sqrt(np.sum(y_pred_mean**2))

        if den == 0:
            return np.nan  # correlation undefined

        return float(num / den)

    except Exception:
        return np.nan

def evaluate(y_true, y_pred):
    """
    Compute a standard set of metrics and return as dict.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
    }
