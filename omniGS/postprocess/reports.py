# omniGS/postprocess/reports.py
import os
import pandas as pd
import numpy as np
import glob

def aggregate_trait_results(trait_file: str):
    """
    Aggregate results for a single trait (mean ± std across replicates/folds).

    """
    df = pd.read_csv(trait_file, sep="\t")
    trait = os.path.splitext(os.path.basename(trait_file))[0]

    grouped = df.groupby("model").agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        pearsonr_mean=("pearsonr", "mean"),
        pearsonr_std=("pearsonr", "std"),
        r2_mean=("model_r2", "mean"),
        r2_std=("model_r2", "std"),
    ).reset_index()

    # Format mean ± std into a single column
    def fmt(mean, std):
        return f"{mean:.4f} ± {std:.4f}"

    grouped["mse"] = grouped.apply(lambda x: fmt(x["mse_mean"], x["mse_std"]), axis=1)
    grouped["rmse"] = grouped.apply(lambda x: fmt(x["rmse_mean"], x["rmse_std"]), axis=1)
    grouped["pearsonr"] = grouped.apply(lambda x: fmt(x["pearsonr_mean"], x["pearsonr_std"]), axis=1)
    grouped["r2"] = grouped.apply(lambda x: fmt(x["r2_mean"], x["r2_std"]), axis=1)

    # Keep only formatted columns + numeric values for ranking
    grouped.insert(0, "trait", trait)
    grouped["pearsonr_val"] = grouped["pearsonr_mean"]
    grouped["r2_val"] = grouped["r2_mean"]

    return grouped[["trait", "model", "mse", "rmse", "pearsonr", "r2",
                    "pearsonr_val", "r2_val"]]


def run_postprocess(geno_dir: str, combine_all: bool = True):
    """
    Run post-processing across all trait TSVs in the phenotypes/ subfolder.
    Save everything into a single Excel file named <geno_base>_summary.xlsx
    inside geno_dir.
    """
    phenotypes_dir = os.path.join(geno_dir, "phenotypes")
    all_raw = []
    trait_summaries = []

    if not os.path.exists(phenotypes_dir):
        raise FileNotFoundError(f"Phenotypes directory not found: {phenotypes_dir}")

    # Extract geno_base from path
    geno_base = os.path.basename(geno_dir.rstrip("/"))
    output_path = os.path.join(geno_dir, f"{geno_base}_summary.xlsx")

    # Collect raw + aggregated summaries
    for f in os.listdir(phenotypes_dir):
        if f.endswith(".tsv") and not f.endswith("_summary.tsv") and f != "summary_all_traits.tsv":
            trait_file = os.path.join(phenotypes_dir, f)

            # Read raw per-fold results
            df_raw = pd.read_csv(trait_file, sep="\t")
            all_raw.append(df_raw)

            # Build aggregated summary
            summary = aggregate_trait_results(trait_file)
            trait_summaries.append(summary)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # First sheet: merged raw results
        if all_raw:
            merged_raw = pd.concat(all_raw, ignore_index=True)
            merged_raw.to_excel(writer, sheet_name="combined_results", index=False)

            # Second sheet: per-trait descriptive statistics
            tidy_stats = []
            for f in os.listdir(phenotypes_dir):
                if f.endswith(".tsv") and not f.endswith("_summary.tsv") and f != "summary_all_traits.tsv":
                    trait_file = os.path.join(phenotypes_dir, f)
                    trait = os.path.splitext(f)[0]

                    df = pd.read_csv(trait_file, sep="\t")
                    grouped = df.groupby("model").describe(percentiles=[0.25, 0.5, 0.75])

                    for model, stats in grouped.groupby(level=0):
                        for metric in ["mse", "rmse", "pearsonr", "model_r2"]:
                            if (metric,) in stats.columns:
                                row = {
                                    "Trait": trait,
                                    "Model": model,
                                    "Metric": metric,
                                    "mean": stats.loc[model, (metric, "mean")],
                                    "std": stats.loc[model, (metric, "std")],
                                    "min": stats.loc[model, (metric, "min")],
                                    "25%": stats.loc[model, (metric, "25%")],
                                    "50%": stats.loc[model, (metric, "50%")],
                                    "75%": stats.loc[model, (metric, "75%")],
                                    "max": stats.loc[model, (metric, "max")],
                                }
                                tidy_stats.append(row)

            summary_df = pd.DataFrame(tidy_stats)
            summary_df.to_excel(writer, sheet_name="summary_statistics", index=False)

            # Third sheet: combined trait summaries (mean ± std)
            if trait_summaries:
                combined_summary = pd.concat(trait_summaries, ignore_index=True)
                combined_summary = combined_summary.sort_values(["trait", "model"])
                combined_summary.drop(columns=["pearsonr_val", "r2_val"], inplace=True)
                combined_summary.to_excel(writer, sheet_name="mean ± std", index=False)

                # Fourth sheet: best model by Pearsonr
                best_by_pearsonr = (
                    pd.concat(trait_summaries, ignore_index=True)
                    .sort_values(["trait", "pearsonr_val"], ascending=[True, False])
                    .groupby("trait")
                    .head(1)
                    .drop(columns=["pearsonr_val", "r2_val"])
                )
                best_by_pearsonr.to_excel(writer, sheet_name="best_by_pearsonr", index=False)

                # Fifth sheet: best model by R2
                best_by_r2 = (
                    pd.concat(trait_summaries, ignore_index=True)
                    .sort_values(["trait", "r2_val"], ascending=[True, False])
                    .groupby("trait")
                    .head(1)
                    .drop(columns=["pearsonr_val", "r2_val"])
                )
                best_by_r2.to_excel(writer, sheet_name="best_by_r2", index=False)

    return output_path

def run_prediction_postprocess(pred_results_dir: str) -> str:
    """
    Collect per-trait prediction outputs and aggregate into a single Excel file.
    Includes both metrics (if available) and averaged predictions.
    """

    excel_path = os.path.join(pred_results_dir, "prediction_summary.xlsx")

    metrics_list, predictions_list = [], []

    # Collect all *_metrics.tsv inside trait folders
    for metrics_file in glob.glob(os.path.join(pred_results_dir, "phenotypes", "*_metrics.tsv")):
        df = pd.read_csv(metrics_file, sep="\t")
        metrics_list.append(df)

    # Collect all *_predictions.tsv at trait level
    for preds_file in glob.glob(os.path.join(pred_results_dir, "phenotypes", "*_predictions.tsv")):
        df = pd.read_csv(preds_file, sep="\t")
        predictions_list.append(df)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if metrics_list:
            all_metrics = pd.concat(metrics_list, ignore_index=True)
            all_metrics.to_excel(writer, sheet_name="Metrics", index=False)

        if predictions_list:
            all_preds = pd.concat(predictions_list, ignore_index=True)
            all_preds.to_excel(writer, sheet_name="Predictions", index=False)

    return excel_path