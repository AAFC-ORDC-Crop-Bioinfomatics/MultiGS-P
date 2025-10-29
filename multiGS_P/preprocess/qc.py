# multiGS_P/preprocess/qc.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# Minor Allele Frequency (MAF)
# -------------------------------

def calc_maf(geno: np.ndarray) -> np.ndarray:
    """
    Calculate Minor Allele Frequency (MAF) for each SNP.

    Args:
        geno (np.ndarray): Genotype matrix (samples x SNPs).
                           Values: 0, 1, 2 (allele counts), -1 (missing).

    Returns:
        np.ndarray: Array of MAF values (length = n_snps).
    """
    n_samples, n_snps = geno.shape
    logger.info(f"Calculating MAF for {n_snps} SNPs across {n_samples} samples")

    maf_values = np.zeros(n_snps)

    for j in range(n_snps):
        snp = geno[:, j]
        snp = snp[snp >= 0]  # remove missing (-1)
        if snp.size == 0:
            maf_values[j] = np.nan
            continue
        allele_freq = np.mean(snp) / 2.0
        maf_values[j] = min(allele_freq, 1 - allele_freq)

    logger.debug(f"First 10 MAF values: {maf_values[:10]}")
    return maf_values


# -------------------------------
# Missingness
# -------------------------------

def calc_missingness(geno: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate missingness (fraction of missing genotypes).

    Args:
        geno (np.ndarray): Genotype matrix (samples x SNPs).
                           Values: 0, 1, 2, -1.
        axis (int, optional): 
            0 = per SNP (default),
            1 = per sample.

    Returns:
        np.ndarray: Missingness values.
    """
    if axis == 0:  # per SNP
        miss = np.mean(geno == -1, axis=0)
        logger.info(f"Calculated missingness per SNP (n={geno.shape[1]})")
        logger.debug(f"First 10 SNP missingness: {miss[:10]}")
        return miss
    elif axis == 1:  # per sample
        miss = np.mean(geno == -1, axis=1)
        logger.info(f"Calculated missingness per sample (n={geno.shape[0]})")
        logger.debug(f"First 10 sample missingness: {miss[:10]}")
        return miss
    else:
        logger.error(f"Invalid axis value: {axis}")
        raise ValueError("axis must be 0 (per SNP) or 1 (per sample)")


# -------------------------------
# Heterozygosity 
# -------------------------------

def calc_heterozygosity(geno: np.ndarray) -> np.ndarray:
    """
    Calculate heterozygosity rate per individual.
    """
    logger.warning("Heterozygosity calculation is not yet implemented")
    raise NotImplementedError("Heterozygosity calculation not yet implemented")


# -------------------------------
# Linkage Disequilibrium 
# -------------------------------

def calc_ld(geno: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Calculate linkage disequilibrium (LD) in a sliding window.
    """
    logger.warning("LD calculation is not yet implemented")
    raise NotImplementedError("LD calculation not yet implemented")