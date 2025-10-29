# multiGS_P/preprocess/loaders.py
import os
import pandas as pd
import allel
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)


def load_vcf(path: str) -> dict:
    """
    Load genotype data and metadata from a VCF file.
    Returns:
        dict with keys:
            geno (np.ndarray): Genotype matrix (n_samples x n_variants).
                               Values: 0, 1, 2 (allele counts), -1 for missing.
            samples (list[str]): List of sample IDs in the same row order as geno.
            variants (pd.DataFrame): Variant metadata with columns:
                CHROM, POS, ID, REF, ALT, QUAL, FILTER_PASS
    """
    if not os.path.exists(path):
        logger.error(f"VCF file not found: {path}")
        raise FileNotFoundError(f"VCF file not found: {path}")

    logger.info(f"Loading raw VCF file: {path}")

    warnings.filterwarnings("ignore", message="'PASS' FILTER header not found")
    warnings.filterwarnings("ignore", message="'GT' FORMAT header not found")
    
    callset = allel.read_vcf(
        path,
        fields=[
            "samples",
            "variants/CHROM",
            "variants/POS",
            "variants/ID",
            "variants/REF",
            "variants/ALT",
            "variants/QUAL",
            "variants/FILTER_PASS",
            "calldata/GT",
        ]
    )

    # Genotype matrix
    gt = allel.GenotypeArray(callset["calldata/GT"])
    geno = gt.to_n_alt(fill=-1).T  # (samples x variants)
    logger.info(f"Genotype matrix loaded: {geno.shape[0]} samples x {geno.shape[1]} variants")

    # Sample IDs
    samples = list(callset["samples"])
    logger.debug(f"First 5 sample IDs: {samples[:5]}")

    # Variant metadata
    variants = pd.DataFrame({
        "CHROM": callset["variants/CHROM"],
        "POS": callset["variants/POS"],
        "ID": callset["variants/ID"],
        "REF": callset["variants/REF"],
        "ALT": [",".join(map(str, a)) for a in callset["variants/ALT"]],
        "QUAL": callset["variants/QUAL"],
        "FILTER_PASS": callset["variants/FILTER_PASS"],
    })
    logger.debug(f"First 5 variants: {variants.head().to_dict(orient='records')}")

    return {
        "geno": geno,
        "samples": samples,
        "variants": variants,
    }


def load_pheno(path: str) -> pd.DataFrame:
    """
    Load phenotype data from a CSV/TSV file.

    Returns:
        pd.DataFrame: Phenotype table (rows = samples, cols = traits).
    """
    if not os.path.exists(path):
        logger.error(f"Phenotype file not found: {path}")
        raise FileNotFoundError(f"Phenotype file not found: {path}")

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    logger.info(f"Loading phenotype file: {path}")

    pheno = pd.read_csv(path, sep=sep)
    logger.info(f"Phenotype loaded: {pheno.shape[0]} samples x {pheno.shape[1] - 1} traits")
    logger.debug(f"First 5 rows:\n{pheno.head()}")

    return pheno