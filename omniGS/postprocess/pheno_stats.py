# omniGS/postprocess/pheno_stats.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm


def run_pheno_analysis(geno_dir: str, pheno_df: pd.DataFrame):
    """
    Perform phenotype analysis: summary statistics, distributions, and trait correlations.

    Args:
        geno_dir (str): Path to genotype-level results directory.
        pheno_df (pd.DataFrame): Phenotype dataframe with columns [SampleID, trait1, trait2, ...].
    """
    outdir = os.path.join(geno_dir, "pheno_analysis")
    os.makedirs(outdir, exist_ok=True)
    dist_dir = os.path.join(outdir, "distributions")
    os.makedirs(dist_dir, exist_ok=True)

    # -------------------------
    # Identify traits
    # -------------------------
    traits = [c for c in pheno_df.columns if c.lower() != "sampleid"]

    # -------------------------
    # Summary statistics
    # -------------------------
    summary = pheno_df[traits].describe().transpose()
    summary.to_csv(os.path.join(outdir, "summary_stats.tsv"), sep="\t")

    # -------------------------
    # Distributions
    # -------------------------
    for trait in traits:
        data = pheno_df[trait].dropna()

        # 1. Histogram (Counts only)
        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=20, color="skyblue", stat="count", kde=False)
        plt.title(f"Distribution of {trait} (Counts)")
        plt.xlabel(trait)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, f"{trait}_hist.png"), dpi=300)
        plt.close()

        # 2. Histogram with Normal Overlay (Density)
        mu, sigma = data.mean(), data.std()

        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=20, color="steelblue", stat="density", kde=False)

        # Overlay theoretical normal distribution
        x = np.linspace(data.min(), data.max(), 100)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, linewidth=2, color="#0D49BD", label=f"N({mu:.2f}, {sigma:.2f}²)")

        plt.title(f"Distribution of {trait} with Normal Overlay (Density)")
        plt.xlabel(trait)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, f"{trait}_hist_normal.png"), dpi=300)
        plt.close()

    # -------------------------
    # Trait-Trait Correlations
    # -------------------------
    corr = pheno_df[traits].corr(method="pearson")
    corr.to_csv(os.path.join(outdir, "trait_correlation.tsv"), sep="\t")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar=True)
    plt.title("Trait-Trait Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "trait_correlation_heatmap.png"), dpi=300)
    plt.close()

    return {
        "summary": os.path.join(outdir, "summary_stats.tsv"),
        "correlation": os.path.join(outdir, "trait_correlation.tsv"),
        "plots_dir": dist_dir
    }


def run_prediction_pheno_analysis(pred_results_dir: str, train_pheno: pd.DataFrame, test_pheno: pd.DataFrame = None):
    """
    Perform phenotype analysis for Prediction mode.
    Runs analysis on train phenotypes and, if available, on test phenotypes.

    Args:
        pred_results_dir (str): Base prediction results directory for this genotype.
        train_pheno (pd.DataFrame): Training phenotype dataframe [SampleID, trait1, ...].
        test_pheno (pd.DataFrame, optional): Test phenotype dataframe [SampleID, trait1, ...].
    """
    outdir = os.path.join(pred_results_dir, "pheno_analysis")
    os.makedirs(outdir, exist_ok=True)

    
    def analyze_pheno(pheno_df: pd.DataFrame, outdir_group: str):
        os.makedirs(outdir_group, exist_ok=True)
        dist_dir = os.path.join(outdir_group, "distributions")
        os.makedirs(dist_dir, exist_ok=True)

        traits = [c for c in pheno_df.columns if c.lower() != "sampleid"]

        # Summary statistics
        summary = pheno_df[traits].describe().transpose()
        summary.to_csv(os.path.join(outdir_group, "summary_stats.tsv"), sep="\t")

        # Trait distributions
        for trait in traits:
            data = pheno_df[trait].dropna()

            # Histogram (Counts)
            plt.figure(figsize=(6, 4))
            sns.histplot(data, bins=20, color="skyblue", stat="count", kde=False)
            plt.title(f"Distribution of {trait} (Counts)")
            plt.xlabel(trait)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f"{trait}_hist.png"), dpi=300)
            plt.close()

            # Histogram + Normal Overlay
            mu, sigma = data.mean(), data.std()
            plt.figure(figsize=(6, 4))
            sns.histplot(data, bins=20, color="steelblue", stat="density", kde=False)
            x = np.linspace(data.min(), data.max(), 100)
            y = norm.pdf(x, mu, sigma)
            plt.plot(x, y, linewidth=2, color="#0D49BD", label=f"N({mu:.2f}, {sigma:.2f}²)")
            plt.title(f"Distribution of {trait} with Normal Overlay (Density)")
            plt.xlabel(trait)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f"{trait}_hist_normal.png"), dpi=300)
            plt.close()

        # Trait-Trait correlations
        corr = pheno_df[traits].corr(method="pearson")
        corr.to_csv(os.path.join(outdir_group, "trait_correlation.tsv"), sep="\t")

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar=True)
        plt.title("Trait-Trait Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_group, "trait_correlation_heatmap.png"), dpi=300)
        plt.close()

    # -------------------------
    # Train phenotype analysis
    # -------------------------
    train_dir = os.path.join(outdir, "train")
    analyze_pheno(train_pheno, train_dir)

    # -------------------------
    # Test phenotype analysis (if provided)
    # -------------------------
    if test_pheno is not None and test_pheno.shape[1] > 1:
        test_dir = os.path.join(outdir, "test")
        analyze_pheno(test_pheno, test_dir)

    return outdir