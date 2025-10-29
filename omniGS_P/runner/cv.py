# multiGS_P/runner/cv.py
import os
import logging
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from multiGS_P.evaluation.splits import make_splits
from multiGS_P.utils.io import write_json, read_json
from multiGS_P.models.registry import get_model_class
from multiGS_P.evaluation.metrics import evaluate


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


def run_trait_cv(trait, geno, pheno, config, geno_base, splits_dir, results_dir, logs_dir):

    logger = setup_trait_logger(trait, logs_dir)

    # Mask samples with missing phenotype for this trait
    mask = ~pheno[trait].isna()
    geno_trait = geno[mask, :]
    pheno_trait = pheno.loc[mask, trait]

    logger.info(
        f"[{trait}] {geno_trait.shape[0]} usable samples "
        f"out of {geno.shape[0]}"
    )

    # Generate trait-specific splits
    splits_path = os.path.join(splits_dir, f"{geno_base}_{trait}_splits.json")

    make_splits(
        n_samples=geno_trait.shape[0],
        n_replicates=config["general"]["n_replicates"],
        n_folds=config["general"]["n_folds"],
        seed=config["general"]["seed"],
        out_path=splits_path,
        trait=trait,
        geno_base=geno_base,
        logger=logger
    )

    root_logger.info(f"Created CV Splits for {trait}")
    # -------------------------
    # Load generated splits
    # -------------------------
    splits = read_json(splits_path, logger)

    all_results = []

    # -------------------------
    # Loop over enabled models
    # -------------------------
    start = time.perf_counter()
    
    for model_name in config["enabled_models"]:
        model_cls = get_model_class(model_name)

        # Get params for this model from config
        params = config["model_params"].get(model_name, {})

        # Pass logger into every model (BRR, RF, CNN, STACKING, etc.)
        if model_name == "STACKING":
            params["logger"] = logger

        logger.info(f"Initializing model {model_name} with params {params}")
        model = model_cls(**params)

        # Iterate through all splits (flat list)
        for split in splits["splits"]:
            rep_idx = split["rep"]
            fold_idx = split["fold"]
            train_idx = np.array(split["train_idx"])
            val_idx = np.array(split["val_idx"])

            X_train, y_train = geno_trait[train_idx], pheno_trait[train_idx]
            X_val, y_val = geno_trait[val_idx], pheno_trait[val_idx]

            # Train
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)

            # Evaluate with metrics.py
            metrics = evaluate(y_val, preds)

            logger.info(
                f"repeat={rep_idx}, fold={fold_idx}, trait={trait}, model={model_name}, "
                f"mse={metrics['rmse']**2:.4f}, rmse={metrics['rmse']:.4f}, "
                f"pearsonr={metrics['pearson']:.4f}, model_r2={metrics['r2']:.4f}"
            )

            # Store results in final format
            all_results.append({
                "repeat": rep_idx,
                "fold": fold_idx,
                "trait": trait,
                "model": model_name,
                "mse": metrics["rmse"] ** 2,
                "rmse": metrics["rmse"],
                "pearsonr": metrics["pearson"],
                "model_r2": metrics["r2"],
            })

    end = time.perf_counter()
    elapsed = end - start
    formatted = str(timedelta(seconds=int(elapsed)))
    logger.info(f"Total Time Taken for all models:{formatted}")
    
    # -------------------------
    # Save results as TSV
    # -------------------------
    cv_out_dir = os.path.join(results_dir, "cross_validation_results", geno_base, "phenotypes")
    os.makedirs(cv_out_dir, exist_ok=True)

    out_path = os.path.join(cv_out_dir, f"{trait}.tsv")
    df = pd.DataFrame(all_results)
    df.to_csv(out_path, sep="\t", index=False)

    logger.info(f"[{trait}] Saved CV results to {out_path}")