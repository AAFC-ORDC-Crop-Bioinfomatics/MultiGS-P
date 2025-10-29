# multiGS_P/evaluation/splits.py
import numpy as np
from sklearn.model_selection import KFold
from multiGS_P.utils.io import write_json
import logging


def make_splits(n_samples, n_replicates, n_folds, seed, out_path: str,
                trait: str = None, geno_base: str = None, logger=None):
    
    if logger is None:
        logger = logging.getLogger(__name__)

    if n_samples < n_folds:
        logger.error(f"Invalid CV setup: n_samples={n_samples} < n_folds={n_folds}")
        raise ValueError("Number of folds cannot exceed number of samples")
    if n_replicates < 1:
        logger.error(f"Invalid n_replicates: {n_replicates}")
        raise ValueError("n_replicates must be >= 1")
    if n_folds < 2:
        logger.error(f"Invalid n_folds: {n_folds}")
        raise ValueError("n_folds must be >= 2")

    rng = np.random.RandomState(seed)
    all_splits = []

    for r in range(n_replicates):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 1e6))
        logger.debug(f"Replicate {r+1}: creating {n_folds} folds")
        for f, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            split = {
                "rep": r + 1,
                "fold": f + 1,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist()
            }
            all_splits.append(split)
            logger.debug(
                f"Rep {r+1}, Fold {f+1}: train={len(train_idx)}, val={len(val_idx)}"
            )


    metadata = {
        "n_samples": n_samples,
        "n_replicates": n_replicates,
        "n_folds": n_folds,
        "seed": seed,
        "splits": all_splits
    }
    if trait:
        metadata["trait"] = trait
    if geno_base:
        metadata["geno_base"] = geno_base

    write_json(metadata, out_path)
    logger.info(
        f"Saved CV splits to {out_path} "
        f"({len(all_splits)} total: {n_replicates} reps × {n_folds} folds)"
    )