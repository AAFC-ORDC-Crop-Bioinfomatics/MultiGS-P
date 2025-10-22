# omniGS/config/parser.py

import configparser
import os


def try_cast(value):
    """
    Cast configuration values into proper Python types.
    """
    value = value.strip().strip('"').strip("'")

    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # None-like
    if value.lower() in ("none", "null", "na"):
        return None

    # List with brackets
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [try_cast(v.strip()) for v in inner.split(",")]

    # Comma-separated list without brackets
    if "," in value:
        return [try_cast(v.strip()) for v in value.split(",")]

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Default: return string
    return value


def parse_config(path):
    """
    Parse an OmniGS configuration (.ini) file into a structured dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    config = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";"),
        interpolation=None
    )
    config.optionxform = str
    config.read(path)

    # --- Section validation ---
    required_sections = ["General", "InputData", "Models", "FeatureView"]
    for sec in required_sections:
        if sec not in config.sections():
            raise ValueError(f"Missing required section: [{sec}]")

    # --- General section ---
    general = {
        "seed": config.getint("General", "seed", fallback=42),
        "n_replicates": config.getint("General", "n_replicates", fallback=1),
        "n_folds": config.getint("General", "n_folds", fallback=5),
        "results_dir": config.get("General", "results_dir", fallback="results"),
        "log_level": config.get("General", "log_level", fallback="INFO").upper(),
        "maf_thresh": config.getfloat("General", "maf_thresh", fallback=0.05),
        "miss_thresh": config.getfloat("General", "miss_thresh", fallback=0.1),
        "n_prediction_repeats": config.getint("General", "n_prediction_repeats", fallback=1),
    }

    if general["seed"] < 0:
        raise ValueError("Seed must be >= 0")
    if general["n_folds"] < 2:
        raise ValueError("n_folds must be >= 2")
    if general["n_replicates"] < 1:
        raise ValueError("n_replicates must be >= 1")

    os.makedirs(general["results_dir"], exist_ok=True)

    # --- Input data section ---
    input_data = {
        "train_geno": config.get("InputData", "train_geno_path", fallback=None),
        "train_pheno": config.get("InputData", "train_pheno_path", fallback=None),
        "test_geno": config.get("InputData", "test_geno_path", fallback=None),
        "test_pheno": config.get("InputData", "test_pheno_path", fallback=None),
    }

    if not input_data["train_geno"] or not os.path.exists(input_data["train_geno"]):
        raise FileNotFoundError(f"Training genotype file not found: {input_data['train_geno']}")
    if not input_data["train_pheno"] or not os.path.exists(input_data["train_pheno"]):
        raise FileNotFoundError(f"Training phenotype file not found: {input_data['train_pheno']}")

    # --- Determine run mode ---
    if input_data["test_geno"] and input_data["test_pheno"]:
        mode = "prediction"
    elif input_data["test_geno"] and not input_data["test_pheno"]:
        mode = "prediction"
    elif not input_data["test_geno"] and input_data["test_pheno"]:
        raise ValueError("Test phenotype provided without test genotype.")
    else:
        mode = "cv"

    if mode == "prediction":
        if input_data["test_geno"] and not os.path.exists(input_data["test_geno"]):
            raise FileNotFoundError(f"Test genotype file not found: {input_data['test_geno']}")
        if input_data["test_pheno"] and not os.path.exists(input_data["test_pheno"]):
            raise FileNotFoundError(f"Test phenotype file not found: {input_data['test_pheno']}")

    # --- Enabled models ---
    truthy = {"true", "1", "yes", "on"}
    enabled_models = [
        m.upper()
        for m, v in config["Models"].items()
        if v.strip().lower() in truthy
    ]
    if not enabled_models:
        raise ValueError("At least one model must be enabled in [Models].")

    # --- Model hyperparameters ---
    model_params = {}
    for section in config.sections():
        if section.startswith("Hyperparameters_"):
            model_name = section.replace("Hyperparameters_", "").upper()
            params = {k: try_cast(v) for k, v in config[section].items()}
            model_params[model_name] = params

            NEGATIVE_ALLOWED = {"max_depth", "num_leaves", "max_iter"}
            for param, val in params.items():
                if isinstance(val, int) and val < 0 and param not in NEGATIVE_ALLOWED:
                    raise ValueError(f"Invalid negative value for {param} in [{section}]")
                if isinstance(val, list) and len(val) == 0:
                    raise ValueError(f"Empty list for {param} in [{section}]")

    for m in enabled_models:
        if m not in model_params:
            raise ValueError(
                f"Enabled model [{m}] missing [Hyperparameters_{m}] section."
            )

    
    if "STACKING" in enabled_models:
        if "STACKING" not in model_params:
            raise ValueError("STACKING=True but [Hyperparameters_STACKING] section missing.")

        params = model_params["STACKING"]

        # Parse base_models
        base_models_raw = params.get("base_models", "")
        base_models = []
        if isinstance(base_models_raw, str):
            base_models = [m.strip().upper() for m in base_models_raw.strip("[]").split(",") if m.strip()]
        elif isinstance(base_models_raw, list):
            base_models = [str(m).strip().upper() for m in base_models_raw]

        if not base_models:
            raise ValueError("STACKING requires base_models to be specified.")
        params["base_models"] = base_models

        # Parse meta_model
        if not params.get("meta_model"):
            raise ValueError("STACKING requires meta_model to be specified.")
        params["meta_model"] = str(params["meta_model"]).strip().upper()

    # --- Feature view ---
    allowed_feature_views = {"SNP", "PC", "HAP"}
    feature_view = config.get("FeatureView", "feature_view", fallback="SNP").upper()
    if feature_view not in allowed_feature_views:
        raise ValueError(f"Invalid feature_view: {feature_view}")

    feature_view_settings = {}
    if "FeatureViewSettings" in config.sections():
        feature_view_settings = {k: try_cast(v) for k, v in config["FeatureViewSettings"].items()}

    if feature_view == "PC":
        comp = feature_view_settings.get("pca_components")
        var_thr = feature_view_settings.get("pca_variance_threshold")
        if comp is None and var_thr is None:
            raise ValueError("For [FeatureView]=PC specify pca_components or pca_variance_threshold.")
        if comp is not None and var_thr is not None:
            raise ValueError("For [FeatureView]=PC specify only one of pca_components or pca_variance_threshold.")

    # --- Return structured config ---
    return {
        "general": general,
        "input": input_data,
        "mode": mode,
        "feature_view": feature_view,
        "feature_view_settings": feature_view_settings,
        "enabled_models": enabled_models,
        "model_params": model_params,
    }