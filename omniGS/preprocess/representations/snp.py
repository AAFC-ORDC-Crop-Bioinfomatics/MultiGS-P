# omniGS/preprocess/representations/snp.py

import numpy as np

def snp_features(geno: np.ndarray) -> np.ndarray:
    """
    Return SNP-based feature representation.
    Since the original input is always SNP, no changes is required.
    """
    return geno
