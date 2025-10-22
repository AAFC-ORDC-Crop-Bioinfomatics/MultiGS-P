# omniGS/preprocess/representations/pca.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_features(geno: np.ndarray, settings: dict):
    """
    Generate PCA features from genotype matrix.

    Args:
        geno (np.ndarray): Genotype matrix of shape (samples, SNPs).
        settings (dict): Parsed FeatureViewSettings from config.
            - pca_components (int, optional): Number of PCs to retain.
            - pca_variance_threshold (float, optional): Variance ratio to retain.

    Returns:
        tuple:
            - X_pca (np.ndarray): PCA-transformed genotype matrix (samples × PCs).
            - explained_variance (pd.DataFrame): Variance explained per PC.
    """
    comp = settings.get("pca_components")
    var_thr = settings.get("pca_variance_threshold")

    if comp is not None:
        pca = PCA(n_components=int(comp), svd_solver="full", random_state=42)
    elif var_thr is not None:
        pca = PCA(n_components=float(var_thr), svd_solver="full", random_state=42)
    else:
        raise ValueError(
            "PCA settings must include either 'pca_components' or 'pca_variance_threshold'."
        )

    X_pca = pca.fit_transform(geno)

    # Explained variance info
    explained_variance = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    return X_pca, explained_variance

def pca_features_prediction(train_geno: np.ndarray, test_geno: np.ndarray, settings: dict):
    """
    PCA for prediction mode:

    Args:
        train_geno (np.ndarray): Training genotype matrix (samples × SNPs).
        test_geno (np.ndarray): Test genotype matrix (samples × SNPs).
        settings (dict): Parsed FeatureViewSettings from config.
            - pca_components (int, optional): Number of PCs to retain.
            - pca_variance_threshold (float, optional): Variance ratio to retain.

    Returns:
        tuple:
            - train_pcs (np.ndarray): PCA-transformed training genotypes (samples × PCs).
            - test_pcs (np.ndarray): PCA-transformed test genotypes (samples × PCs).
            - explained_variance (pd.DataFrame): Variance explained per PC.
    """
    from sklearn.decomposition import PCA

    comp = settings.get("pca_components")
    var_thr = settings.get("pca_variance_threshold")

    if comp is not None:
        pca = PCA(n_components=int(comp), svd_solver="full", random_state=42)
    elif var_thr is not None:
        pca = PCA(n_components=float(var_thr), svd_solver="full", random_state=42)
    else:
        raise ValueError(
            "PCA settings must include either 'pca_components' or 'pca_variance_threshold'."
        )

    
    train_pcs = pca.fit_transform(train_geno)
    test_pcs = pca.transform(test_geno)

    # Explained variance info
    explained_variance = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    return train_pcs, test_pcs, explained_variance


def plot_pca_variance(explained_variance: pd.DataFrame, out_path: str, threshold: float = None):
    """
    Plot cumulative variance explained by PCA components.

    Args:
        explained_variance (pd.DataFrame): Output from pca_features
        out_path (str): Path to save PNG
        threshold (float, optional): Draw horizontal cutoff line (e.g., 0.95).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(explained_variance) + 1),
        explained_variance["cumulative_variance"],
        marker="o"
    )
    if threshold:
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
        plt.legend()

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("PCA Cumulative Variance Explained")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()