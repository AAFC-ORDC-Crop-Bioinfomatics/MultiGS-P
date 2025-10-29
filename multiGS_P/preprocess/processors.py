# multiGS_P/preprocess/processors.py

import numpy as np
import pandas as pd
import logging
from .qc import calc_maf, calc_missingness

logger = logging.getLogger(__name__)

# -------------------------------
# SNP Filtering
# -------------------------------

def filter_snps(geno: np.ndarray, maf_thresh: float = 0.05, miss_thresh: float = 0.1) -> np.ndarray:
    """
    Filter SNPs based on MAF and missingness thresholds.
    Args:
        geno (np.ndarray): Genotype matrix (samples x SNPs).
                           Values: 0, 1, 2 (allele counts), -1 (missing).
        maf_thresh (float, optional): Minimum allowed MAF. Defaults to 0.05.
        miss_thresh (float, optional): Maximum allowed missing fraction. Defaults to 0.1.

    Returns:
        np.ndarray: Filtered genotype matrix (samples x kept_snps).
    """
    n_samples, n_snps = geno.shape
    logger.info(f"Filtering {n_snps} SNPs across {n_samples} samples "
                f"(MAF ≥ {maf_thresh}, Missing ≤ {miss_thresh})")

    maf = calc_maf(geno)
    miss = calc_missingness(geno, axis=0)
    keep = (maf >= maf_thresh) & (miss <= miss_thresh)
    n_kept = np.sum(keep)

    logger.info(f"SNPs kept: {n_kept}/{n_snps} ({n_kept/n_snps:.2%})")
    return geno[:, keep]


# -------------------------------
# Genotype–Phenotype Alignment
# -------------------------------

def align_geno_pheno(geno: np.ndarray, sample_ids: list[str], pheno: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Align genotype matrix with phenotype samples.

    Args:
        geno (np.ndarray): Genotype matrix (n_samples x n_snps).
        sample_ids (list[str]): Sample names from VCF or cache,
                                in same row order as geno.
        pheno (pd.DataFrame): Phenotype table where:
                              - First column = sample names
                              - Remaining columns = traits.

    Returns:
        tuple: (geno_aligned, pheno_aligned)
            geno_aligned (np.ndarray): Genotype rows reordered to match phenotype.
            pheno_aligned (pd.DataFrame): Phenotype rows reordered and trimmed
                                          to intersection with genotype samples.
    """
    pheno_sample_col = pheno.columns[0]
    pheno = pheno.rename(columns={pheno_sample_col: "SampleID"})
    pheno = pheno.set_index("SampleID")

    common_ids = [sid for sid in pheno.index if sid in sample_ids]
    if not common_ids:
        logger.error("No overlapping samples between genotype and phenotype.")
        raise ValueError("No overlapping samples between genotype and phenotype.")

    logger.info(f"Aligning genotype and phenotype: "
                f"{len(common_ids)} common samples found "
                f"(geno={len(sample_ids)}, pheno={len(pheno)})")

    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    geno_mask = [id_to_idx[sid] for sid in common_ids]

    geno_aligned = geno[geno_mask, :]
    pheno_aligned = pheno.loc[common_ids]

    logger.debug(f"First 5 aligned samples: {common_ids[:5]}")
    return geno_aligned, pheno_aligned.reset_index()


# -------------------------------
# Missing Data Imputation
# -------------------------------

def impute_missing(geno: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Impute missing genotype values.

    Args:
        geno (np.ndarray): Genotype matrix (samples x SNPs).
        method (str, optional): Imputation method.
                                - "mean": fill with SNP mean
                                - "mode": fill with SNP mode
                                Defaults to "mean".

    Returns:
        np.ndarray: Genotype matrix with missing values imputed.
    """
    X = geno.copy()
    n_samples, n_snps = X.shape
    logger.info(f"Imputing missing genotypes using method='{method}' "
                f"for {n_snps} SNPs across {n_samples} samples")

    for j in range(n_snps):
        snp = X[:, j]
        missing_mask = (snp == -1)
        if not np.any(missing_mask):
            continue

        if method == "mean":
            fill_value = np.nanmean(np.where(snp == -1, np.nan, snp))
        elif method == "mode":
            vals, counts = np.unique(snp[snp >= 0], return_counts=True)
            fill_value = vals[np.argmax(counts)]
        else:
            logger.error(f"Invalid imputation method: {method}")
            raise ValueError("method must be 'mean' or 'mode'")

        X[missing_mask, j] = fill_value

    return X


# -------------------------------
# Normalization
# -------------------------------

def normalize(geno: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize genotype matrix.

    Args:
        geno (np.ndarray): Genotype matrix (samples x SNPs).
        method (str, optional): Normalization method.
                                - "zscore": standardize per SNP
                                - "none": no normalization
                                Defaults to "zscore".

    Returns:
        np.ndarray: Normalized genotype matrix.
    """
    n_samples, n_snps = geno.shape
    if method == "none":
        logger.info(f"No normalization applied (samples={n_samples}, SNPs={n_snps})")
        return geno

    elif method == "zscore":
        logger.info(f"Applying z-score normalization "
                    f"(samples={n_samples}, SNPs={n_snps})")
        mean = np.mean(geno, axis=0)
        std = np.std(geno, axis=0)
        std[std == 0] = 1.0
        return (geno - mean) / std

    else:
        logger.error(f"Invalid normalization method: {method}")
        raise ValueError("method must be 'zscore' or 'none'")


def impute_and_fit_transform(geno: np.ndarray, method: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """
    Impute missing genotype values (fit) and return the imputation parameters (fill_values).
    Used for the training data.
    """
    X = geno.copy()
    n_samples, n_snps = X.shape
    fill_values = np.zeros(n_snps)

    for j in range(n_snps):
        snp = X[:, j]
        missing_mask = (snp == -1)
        
        if method == "mean":
            # Calculate mean ignoring the missing values (-1)
            fill_values[j] = np.nanmean(np.where(snp == -1, np.nan, snp))
        elif method == "mode":
            # Calculate mode ignoring the missing values (-1)
            vals, counts = np.unique(snp[snp >= 0], return_counts=True)
            if vals.size > 0:
                fill_values[j] = vals[np.argmax(counts)]
            else:
                fill_values[j] = 0.0 # Default to 0 if no non-missing values
        else:
            raise ValueError(f"Invalid imputation method: {method}. Must be 'mean' or 'mode'")
        
        X[missing_mask, j] = fill_values[j]
    
    return X, fill_values

def impute_transform(geno: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """
    Apply pre-calculated imputation values (from training data) to new data.
    Used for the test data.
    """
    X = geno.copy()
    if X.shape[1] != fill_values.size:
         logger.error("Imputation failed: Genotype columns do not match fill_values size.")
         return X # Return un-imputed array to prevent crash
         
    for j in range(X.shape[1]):
        missing_mask = (X[:, j] == -1)
        X[missing_mask, j] = fill_values[j]
    
    return X

# --- Normalization Utilities ---

def normalize_and_fit_transform(geno: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize genotype matrix (fit) and return the mean and std dev.
    Used for the training data.
    """
    mean = np.mean(geno, axis=0)
    std = np.std(geno, axis=0)
    # Prevent division by zero for monomorphic markers
    std[std == 0] = 1.0 
    normalized_geno = (geno - mean) / std
    
    return normalized_geno, mean, std

def normalize_transform(geno: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply pre-calculated mean and std dev (from training data) to new data.
    Used for the test data.
    """
    # Use the stored mean and std, ensuring std is not zero
    std[std == 0] = 1.0 
    return (geno - mean) / std

# --- Variant Intersection Utility ---

def intersect_and_order_snps(
    train_variants: pd.DataFrame, test_variants: pd.DataFrame,
    train_geno: np.ndarray, test_geno: np.ndarray
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Identifies common variants between train and test sets, and reorders matrices
    based on the variant order in the training set.
    """
    logger.info("Identifying common variants based on CHROM and POS.")
    
    # Create unique identifier for each variant
    train_variants['var_id'] = train_variants['CHROM'].astype(str) + '_' + train_variants['POS'].astype(str)
    test_variants['var_id'] = test_variants['CHROM'].astype(str) + '_' + test_variants['POS'].astype(str)
    
    # Find common IDs
    common_var_ids = np.intersect1d(train_variants['var_id'].values, test_variants['var_id'].values)
    if common_var_ids.size == 0:
        logger.error("No overlapping variants found between training and testing data!")
        raise ValueError("No overlapping variants found.")

    logger.info(f"Overlapping variants: {common_var_ids.size}")

    # Create masks to select common variants in both matrices
    train_mask = train_variants['var_id'].isin(common_var_ids)
    test_mask_original = test_variants['var_id'].isin(common_var_ids)
    
    train_variants_common = train_variants[train_mask]
    test_variants_common = test_variants[test_mask_original]
    
    # Reorder test variants to match the order of train variants
    train_id_to_idx = {vid: i for i, vid in enumerate(train_variants_common['var_id'].values)}
    test_id_to_idx = {vid: i for i, vid in enumerate(test_variants_common['var_id'].values)}
    
    # Get indices for reordering
    train_indices = train_variants[train_mask].index.values
    test_indices = test_variants[test_mask_original].index.values
    
    # Now order the test indices according to the train order
    ordered_test_indices = np.array([
        test_indices[test_id_to_idx[vid]] 
        for vid in train_variants_common['var_id'].values 
        if vid in test_id_to_idx
    ])
    
    # Apply masking and ordering
    train_geno_ordered = train_geno[:, train_indices]
    test_geno_ordered = test_geno[:, ordered_test_indices]
    
    # Return the variant metadata (reordered to match the genotype matrices)
    common_variants_df = train_variants_common.set_index('var_id').loc[common_var_ids].reset_index()

    return train_geno_ordered, test_geno_ordered, common_variants_df

def qc_filter_fit_transform(
    train_geno: np.ndarray, test_geno: np.ndarray, 
    maf_thresh: float, miss_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies QC filters (MAF and missingness) fit only on the training data.
    The resulting QC mask is then applied to both training and test genotypes.
    """
    logger.info("Applying QC filters (fit on TRAIN).")
    
    # Calculate QC metrics on training data only
    maf_values = calc_maf(train_geno)
    miss_values = calc_missingness(train_geno, axis=0)
    
    # Determine which SNPs pass QC
    qc_mask = (maf_values >= maf_thresh) & (miss_values <= miss_thresh)
    n_kept = np.sum(qc_mask)

    logger.info(f"SNPs kept after QC: {n_kept}/{train_geno.shape[1]}")
    
    # Apply the mask to both training and test data
    train_geno = train_geno[:, qc_mask]
    test_geno = test_geno[:, qc_mask]

    return train_geno, test_geno