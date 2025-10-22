# omniGS/runner/prediction.py
import os
import logging
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from omniGS.models.registry import get_model_class
from omniGS.evaluation.metrics import evaluate

root_logger = logging.getLogger(__name__)


def setup_trait_logger(trait: str, logs_dir: str):
    """
    Each trait log goes to results/logs/<trait>.log
    """
    log_path = os.path.join(logs_dir, f"{trait}.log")

    logger = logging.getLogger(trait)
    logger.setLevel(logging.INFO)

    
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s(): %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    
    logger.propagate = False

    return logger


def run_trait_prediction(
    trait,
    train_geno,
    train_pheno,
    test_geno,
    test_pheno,
    config,
    geno_base,
    results_dir,
    logs_dir
):
    
    logger = setup_trait_logger(trait, logs_dir)

    # Config param
    n_repeats = int(config["general"].get("n_prediction_repeats", 1))

    # -------------------------
    # Train masking
    # -------------------------
    mask = ~train_pheno[trait].isna()
    X_train = train_geno[mask, :]
    y_train = train_pheno.loc[mask, trait].values

    logger.info(f"[{trait}] {X_train.shape[0]} usable training samples out of {train_geno.shape[0]}")
    X_test = test_geno

    # -------------------------
    # Test phenotype handling
    # -------------------------
    y_test = None
    eval_mask = None
    if not test_pheno.empty and trait in test_pheno.columns:
        y_test = test_pheno[trait].values
        sample_ids = test_pheno["SampleID"].values
        eval_mask = ~pd.isna(y_test)
        logger.info(
            f"[{trait}] {X_test.shape[0]} test samples provided "
            f"({np.sum(eval_mask)} usable for evaluation)"
        )
    else:
        sample_ids = np.arange(len(X_test))
        logger.warning(f"[{trait}] No test phenotypes provided. Predictions will not be evaluated.")


    preds_runs = {m: [] for m in config["enabled_models"]}

    # Directory for per-run outputs
    trait_dir = os.path.join(results_dir, "prediction_results", geno_base, "phenotypes", trait)
    os.makedirs(trait_dir, exist_ok=True)

    start = time.perf_counter()

    # -------------------------
    # Repeat training
    # -------------------------
    for repeat in range(n_repeats):
        logger.info(f"[{trait}] Starting repeat {repeat+1}/{n_repeats}")

        run_preds = {"SampleID": sample_ids}
        if y_test is not None:
            run_preds["true_value"] = y_test

        for model_name in config["enabled_models"]:
            model_cls = get_model_class(model_name)

            # Get params from config
            params = config["model_params"].get(model_name, {})

            # Pass logger into all models (BRR, RF, CNN, STACKING, etc.)
            if model_name == "STACKING":
                params["logger"] = logger

            logger.info(f"[{trait}] Repeat {repeat+1}: Training model {model_name} with params {params}")
            model = model_cls(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            preds = np.ravel(preds)
            preds_runs[model_name].append(preds)

            run_preds[f"pred_{model_name}"] = preds

        # Save this run’s predictions
        run_df = pd.DataFrame(run_preds)
        run_out_path = os.path.join(trait_dir, f"{trait}_run{repeat+1}.tsv")
        run_df.to_csv(run_out_path, sep="\t", index=False)
        logger.info(f"[{trait}] Saved predictions for repeat {repeat+1} -> {run_out_path}")


    # Aggregate across runs

    preds_out = {}
    for model_name, all_runs in preds_runs.items():
        all_runs = np.vstack(all_runs)  # shape: (n_repeats, n_samples)
        preds_out[model_name] = {
            "mean": np.mean(all_runs, axis=0),
            "std": np.std(all_runs, axis=0),
        }


    # Save averaged predictions

    pred_out_dir = os.path.join(results_dir, "prediction_results", geno_base, "phenotypes")
    os.makedirs(pred_out_dir, exist_ok=True)

    out_path_preds = os.path.join(pred_out_dir, f"{trait}_predictions.tsv")
    if y_test is not None:
        df_preds = pd.DataFrame({"SampleID": sample_ids, "true_value": y_test})
        for model_name, vals in preds_out.items():
            df_preds[f"pred_{model_name}"] = vals["mean"]
            df_preds[f"pred_{model_name}_std"] = vals["std"]
    else:
        df_preds = pd.DataFrame({"SampleID": sample_ids})
        for model_name, vals in preds_out.items():
            df_preds[f"pred_{model_name}"] = vals["mean"]
            df_preds[f"pred_{model_name}_std"] = vals["std"]

    df_preds.to_csv(out_path_preds, sep="\t", index=False)
    logger.info(f"[{trait}] Saved averaged predictions -> {out_path_preds}")

    # -------------------------
    # Metrics (only if phenotypes exist)
    # -------------------------
    if y_test is not None:
        all_results = []
        for model_name, vals in preds_out.items():
            if eval_mask is not None and np.any(eval_mask):
                metrics = evaluate(y_test[eval_mask], vals["mean"][eval_mask])
                all_results.append({
                    "trait": trait,
                    "model": model_name,
                    "mse": metrics["rmse"] ** 2,
                    "rmse": metrics["rmse"],
                    "pearsonr": metrics["pearson"],
                })

        if len(all_results) > 0:
            out_path_metrics = os.path.join(pred_out_dir, f"{trait}_metrics.tsv")
            pd.DataFrame(all_results).to_csv(out_path_metrics, sep="\t", index=False)
            logger.info(f"[{trait}] Saved prediction metrics -> {out_path_metrics}")

    end = time.perf_counter()
    formatted = str(timedelta(seconds=int(end - start)))
    logger.info(f"[{trait}] Completed all repeats. Time taken: {formatted}")