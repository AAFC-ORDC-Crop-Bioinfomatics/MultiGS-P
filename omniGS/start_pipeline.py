# omniGS/start_pipeline.py

import logging
import os
import time
import sys
from datetime import timedelta
from omniGS.cli import parse_cli
from omniGS.config.parser import parse_config
from omniGS.utils.logging import setup_logging
from multiprocessing import Process

# Runner modules
from omniGS.runner.cv import run_trait_cv
from omniGS.runner.prediction import run_trait_prediction

# Postprocess modules
from omniGS.postprocess.reports import run_postprocess, run_prediction_postprocess
from omniGS.postprocess.stats import run_stats
from omniGS.postprocess.pheno_stats import run_pheno_analysis, run_prediction_pheno_analysis
from omniGS.postprocess.plots import run_plots, run_prediction_plots, run_prediction_mds_plot

# Preprocess modules
from omniGS.preprocess.pipeline import run_preprocessing_cv, run_preprocessing_prediction

import warnings
warnings.filterwarnings("ignore")


def start_pipeline(config_path: str = None, args=None):
    """
    Launch the OmniGS pipeline for Cross-Validation or Prediction modes.

    Args:
        config_path (str): Path to the configuration file (.ini).
        args: Optional CLI arguments (used when called from command line).
    """

    # --- Step 1: Parse configuration ---
    if config_path is None:
        cli_args = parse_cli() if args is None else args
        config_file = cli_args.config
    else:
        config_file = config_path

    try:
        config = parse_config(config_file)
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    geno_dir = os.path.dirname(config["input"]["train_geno"])
    geno_base = os.path.splitext(os.path.basename(config["input"]["train_geno"]))[0]

    # --- Step 2: Initialize logging ---
    log_path = setup_logging(
        config["general"]["results_dir"],
        config["mode"],
        geno_base,
        config["general"]["log_level"]
    )

    # --- Step 3: Display run summary ---
    logging.info("=" * 60)
    logging.info("OmniGS Pipeline – Run")
    logging.info("=" * 60)
    logging.info(f"Config file: {config_file}")
    logging.info(f"Logging to: {log_path}")
    logging.info(f"Seed: {config['general']['seed']}")
    logging.info(f"Mode: {config['mode'].upper()}")

    if config["mode"].lower() == "cv":
        logging.info(f"n_replicates: {config['general']['n_replicates']}")
        logging.info(f"n_folds: {config['general']['n_folds']}")
    elif config["mode"].lower() == "prediction":
        logging.info(f"n_prediction_repeats: {config['general']['n_prediction_repeats']}")

    logging.info(f"Results dir: {config['general']['results_dir']}")
    logging.info("-" * 60)

    # Input data summary
    logging.info("Input Data:")
    for k, v in config["input"].items():
        logging.info(f"  {k}: {v}")
    logging.info("-" * 60)

    # Enabled models
    logging.info("Enabled Models:")
    for m in config["enabled_models"]:
        params = config["model_params"].get(m, {})
        logging.info(f"  {m} | Params = {params}")
    logging.info("-" * 60)

    # Feature view
    logging.info(f"Feature View: {config['feature_view']}")
    logging.info("-" * 60)

    # --- Step 4: Run pipeline based on mode ---
    if config["mode"] == "cv":
        logging.info("Running preprocessing for Cross-Validation mode.")
        geno, pheno = run_preprocessing_cv(config)
        logging.info(
            f"Preprocessing complete. Genotype shape: {geno.shape}, "
            f"Phenotype shape: ({pheno.shape[0]}, {pheno.shape[1]-1})"
        )

        # Output directories
        splits_dir = os.path.join(geno_dir, "omnigs-splits")
        results_dir = config["general"]["results_dir"]
        logs_dir = os.path.join(results_dir, "cross_validation_results", geno_base, "logs")

        os.makedirs(splits_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Model training per trait (parallel)
        logging.info("Model training begins. Check individual logs for each trait.")
        start = time.perf_counter()
        processes = []
        for trait in pheno.columns[1:]:  # Skip SampleID
            p = Process(
                target=run_trait_cv,
                args=(trait, geno, pheno, config, geno_base, splits_dir, results_dir, logs_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.perf_counter()
        elapsed = str(timedelta(seconds=int(end - start)))
        logging.info(f"Cross-Validation completed. Time taken: {elapsed}")

        # Postprocessing
        geno_dir_results = os.path.join(results_dir, "cross_validation_results", geno_base)
        logging.info(f"Starting post-processing for genotype: {geno_base}")
        excel_path = run_postprocess(geno_dir_results, combine_all=True)
        logging.info(f"Summary saved to {excel_path}")

        logging.info("Running ANOVA + Tukey tests.")
        run_stats(geno_dir_results)

        logging.info("Running phenotype analysis.")
        run_pheno_analysis(geno_dir_results, pheno)

        logging.info("Generating plots.")
        run_plots(geno_dir_results)

    elif config["mode"] == "prediction":
        logging.info("Running preprocessing for Prediction mode.")
        train_geno, train_pheno, test_geno, test_pheno = run_preprocessing_prediction(config)
        logging.info(
            f"Preprocessing complete.\n"
            f"Train Genotype shape: {train_geno.shape}, Train Phenotype shape: ({train_pheno.shape[0]}, {train_pheno.shape[1]-1})\n"
            f"Test Genotype shape: {test_geno.shape}, Test Phenotype shape: ({test_pheno.shape[0]}, {test_pheno.shape[1]-1})"
        )

        # Output directories
        results_dir = config["general"]["results_dir"]
        logs_dir = os.path.join(results_dir, "prediction_results", geno_base, "logs")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Prediction per trait (parallel)
        logging.info("Prediction begins. Running inference for each trait.")
        start = time.perf_counter()
        processes = []
        for trait in train_pheno.columns[1:]:  # Skip SampleID
            p = Process(
                target=run_trait_prediction,
                args=(trait, train_geno, train_pheno, test_geno, test_pheno,
                      config, geno_base, results_dir, logs_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.perf_counter()
        formatted = str(timedelta(seconds=int(end - start)))
        logging.info(f"Prediction completed. Time taken: {formatted}")

        # Postprocessing
        pred_dir_results = os.path.join(results_dir, "prediction_results", geno_base)

        logging.info(f"Starting post-processing for genotype: {geno_base}")
        excel_path = run_prediction_postprocess(pred_dir_results)
        logging.info(f"Summary saved to {excel_path}")

        logging.info("Running phenotype analysis.")
        run_prediction_pheno_analysis(pred_dir_results, train_pheno, test_pheno)

        logging.info("Generating prediction plots.")
        run_prediction_plots(pred_dir_results)

        logging.info("Generating MDS scatter plot.")
        train_samples = list(train_pheno["SampleID"]) if "SampleID" in train_pheno.columns else []
        test_samples = list(test_pheno["SampleID"]) if (test_pheno is not None and "SampleID" in test_pheno.columns) else []
        mds_outputs = run_prediction_mds_plot(pred_dir_results, train_geno, test_geno, train_samples, test_samples)

        logging.info(f"MDS coordinates saved to {mds_outputs['coords']}")
        logging.info(f"MDS scatter plot saved to {mds_outputs['plot']}")

    else:
        raise ValueError(f"Unknown mode: {config['mode']}")

    logging.info("=" * 60)