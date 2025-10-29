# multiGS_P/postprocess/plots.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import MDS
from numpy.linalg import eigh

def run_plots(geno_dir: str):
    """
    Generate plots per trait across models:
    - Single metric plots (RMSE, Pearsonr, R2)
    - Subplots (Pearsonr & R2 only)
    """
    pheno_dir = os.path.join(geno_dir, "phenotypes")
    if not os.path.exists(pheno_dir):
        raise FileNotFoundError(f"Phenotypes folder not found: {pheno_dir}")

    # Output dirs
    outdir = os.path.join(geno_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    subplot_dir = os.path.join(outdir, "subplots")
    os.makedirs(subplot_dir, exist_ok=True)

    metrics = ["rmse", "pearsonr", "model_r2"]

    # Loop over all trait result files
    for f in os.listdir(pheno_dir):
        if f.endswith(".tsv") and not f.endswith("_summary.tsv") and f != "summary_all_traits.tsv":
            trait_file = os.path.join(pheno_dir, f)
            trait = os.path.splitext(f)[0]

            df = pd.read_csv(trait_file, sep="\t")

            # -------------------------
            # Single metric plots
            # -------------------------
            metric_labels = {
                "rmse": "Root Mean Squared Error (RMSE)",
                "pearsonr": "Prediction accuracy (r)",
                "model_r2": "Coefficient of Determination (R²)"
            }

            for metric in metrics:
                if metric not in df.columns:
                    continue
            
                plt.figure(figsize=(8, 6))
                sns.boxplot(x="model", y=metric, hue="model", data=df, palette="Set2", legend=False)
            
                # Use friendly labels
                ylabel = metric_labels.get(metric, metric)
                plt.xlabel("Model")
                plt.ylabel(ylabel)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
            
                plot_path = os.path.join(outdir, f"{trait}_{metric}.png")
                plt.savefig(plot_path, dpi=300)
                plt.close()

            # -------------------------
            # Subplots: Pearsonr & R2
            # -------------------------
            if "pearsonr" in df.columns and "model_r2" in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

                # Pearsonr subplot
                sns.boxplot(x="model", y="pearsonr", hue="model", data=df, palette="Set2", legend=False, ax=axes[0])
                axes[0].set_title(f"Pearsonr")
                axes[0].set_xlabel("Model")
                axes[0].set_ylabel("Prediction accuracy (r)")
                axes[0].tick_params(axis="x", rotation=30)

                # R2 subplot
                sns.boxplot(x="model", y="model_r2", hue="model", data=df, palette="Set2", legend=False, ax=axes[1])
                axes[1].set_title(f"R²")
                axes[1].set_xlabel("Model")
                axes[1].set_ylabel("R²")
                axes[1].tick_params(axis="x", rotation=30)

                plt.suptitle(f"{trait}", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                subplot_path = os.path.join(subplot_dir, f"{trait}_pearsonr_r2.png")
                plt.savefig(subplot_path, dpi=300)
                plt.close()


def run_prediction_plots(pred_results_dir: str):
    """
    Generate plots for Prediction mode.
    - Scatter plots: true vs predicted with best-fit line
      (True values = blue, Predicted values = yellow)
    - Bar plots: Pearson r and R² across models with values on top
    """
    plots_dir = os.path.join(pred_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # -------------------------
    # Scatter plots (true vs predicted)
    # -------------------------
    pred_files = glob.glob(os.path.join(pred_results_dir, "phenotypes", "*_predictions.tsv"))
    for preds_file in pred_files:
        trait = os.path.basename(preds_file).replace("_predictions.tsv", "")
        df = pd.read_csv(preds_file, sep="\t")

        if "true_value" not in df.columns:
            continue  # skip if no ground truth (no test_pheno)

        for col in df.columns:
            if col.startswith("pred_") and not col.endswith("_std"):
                model = col.replace("pred_", "")

                plt.figure(figsize=(6, 6))

                # True values (blue diagonal)
                plt.scatter(
                    df["true_value"], df["true_value"],
                    alpha=0.6, color="blue", label="True values"
                )

                # Predicted values (yellow)
                plt.scatter(
                    df["true_value"], df[col],
                    alpha=0.7, color="yellow", edgecolor="black", label="Predicted values"
                )

                # Best-fit regression line (black dashed)
                m, b = np.polyfit(df["true_value"], df[col], 1)
                x_vals = np.linspace(df["true_value"].min(), df["true_value"].max(), 100)
                plt.plot(x_vals, m * x_vals + b, "k--", linewidth=2)

                plt.xlabel("True Value")
                plt.ylabel("Predicted Value")
                plt.title(f"{trait} – {model}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{trait}_{model}_scatter.png"), dpi=300)
                plt.close()

    # -------------------------
    # Barplots
    # -------------------------
    metrics_files = glob.glob(os.path.join(pred_results_dir, "phenotypes", "*_metrics.tsv"))
    if metrics_files:
        all_metrics = pd.concat([pd.read_csv(f, sep="\t") for f in metrics_files], ignore_index=True)

        for trait, trait_df in all_metrics.groupby("trait"):
            # Pearson r barplot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="model", y="pearsonr", hue="model", data=trait_df, palette="Blues", legend=False)
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=10, color="black",
                    xytext=(0, 3), textcoords="offset points"
                )
            plt.title(f"{trait} – Pearson Correlation by Model")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Pearson r")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{trait}_pearson_barplot.png"), dpi=300)
            plt.close()

    return plots_dir


def run_prediction_mds_plot(pred_results_dir: str,
                            train_geno: np.ndarray,
                            test_geno: np.ndarray,
                            train_samples: list[str],
                            test_samples: list[str] = None):
    """
    Generate GRM-based classical MDS scatter plot showing genetic relationship
    between train and test samples. Mirrors the logic of plot_mds_from_grm().

    Args:
        pred_results_dir (str): Output directory for prediction results.
        train_geno (np.ndarray): Training genotype matrix (samples × markers).
        test_geno (np.ndarray): Test genotype matrix (samples × markers).
        train_samples (list[str]): Training sample IDs.
        test_samples (list[str], optional): Test sample IDs. Defaults to None.
    """


    if test_samples is None or len(test_samples) != test_geno.shape[0]:
        test_samples = [f"NA_{i+1}" for i in range(test_geno.shape[0])]

    samples = list(train_samples) + list(test_samples)
    groups = (["Train"] * train_geno.shape[0]) + (["Test"] * test_geno.shape[0])


    X = np.vstack([train_geno, test_geno])


    X_centered = X - np.nanmean(X, axis=0)
    stds = np.nanstd(X_centered, axis=0)
    stds[stds == 0] = 1.0 
    X_std = X_centered / stds

 
    M = X_std.shape[1]
    G = np.dot(X_std, X_std.T) / M

    
    diag = np.diag(G)
    D2 = diag[:, None] + diag[None, :] - 2.0 * G

    
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)

    vals, vecs = eigh(B)
    idx = np.argsort(vals)[::-1]  
    vals = vals[idx]
    vecs = vecs[:, idx]

    
    k = 2
    coords = vecs[:, :k] * np.sqrt(np.maximum(vals[:k], 0))

   
    df = pd.DataFrame({
        "SampleID": samples,
        "Group": groups,
        "MDS1": coords[:, 0],
        "MDS2": coords[:, 1]
    })

    os.makedirs(pred_results_dir, exist_ok=True)
    coords_path = os.path.join(pred_results_dir, "MDS_coordinates_GRM.tsv")
    plot_path = os.path.join(pred_results_dir, "MDS_scatter_GRM.png")
    df.to_csv(coords_path, sep="\t", index=False)

    
    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(
        coords[:train_geno.shape[0], 0],
        coords[:train_geno.shape[0], 1],
        label="Train", alpha=0.7
    )
    plt.scatter(
        coords[train_geno.shape[0]:, 0],
        coords[train_geno.shape[0]:, 1],
        label="Test", alpha=0.7
    )
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.title("MDS (GRM-based) – Train vs Test Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


    return {
        "coords": coords_path,
        "plot": plot_path
    }
