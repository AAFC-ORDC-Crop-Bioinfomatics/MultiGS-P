# omniGS_P/preprocess/representations/haplotype.py

import os
import subprocess
import logging
import numpy as np
import pandas as pd
from omniGS_P.preprocess.loaders import load_vcf
from omniGS_P.preprocess.processors import intersect_and_order_snps

logger = logging.getLogger(__name__)

def filter_biallelic_pred(data: dict) -> dict:
    """
    keep only biallelic SNPs.

    Args:
        data (dict): Output of load_vcf with keys 'geno', 'samples', 'variants'

    Returns:
        dict with filtered geno, samples, variants
    """
    variants = data["variants"]
    geno = data["geno"]

    def is_biallelic_row(ref, alt):
        # ALT may be comma-separated like "A" or "G,T"
        alts = [a for a in str(alt).split(",") if a]
        return len(ref) == 1 and len(alts) == 1 and len(alts[0]) == 1

    mask = variants.apply(lambda row: is_biallelic_row(row["REF"], row["ALT"]), axis=1)

    filtered_variants = variants[mask].reset_index(drop=True)
    filtered_geno = geno[:, mask.values]  # keep SNPs only

    logger.info(f"Biallelic filter: kept {filtered_geno.shape[1]} / {geno.shape[1]} variants")

    return {
        "geno": filtered_geno,
        "samples": data["samples"],
        "variants": filtered_variants,
    }

def apply_variant_mask(data: dict, mask: pd.Series):
    """filter variants and genotype matrix."""
    geno = data["geno"]
    variants = data["variants"]
    
    filtered_variants = variants[mask].reset_index(drop=True)
    # Ensure variants are filtered in the genotype matrix (n_samples x n_variants)
    filtered_geno = geno[:, mask.values] 
    
    return {
        "geno": filtered_geno,
        "samples": data["samples"],
        "variants": filtered_variants,
    }

def intersect_variants_by_pos(data1: dict, data2: dict):
    """
    Finds and keeps only the variants (SNPs) that are present in both VCF files 
    based on matching CHROM and POS. This extracts "common SNPs".
    
    Args:
        data1 (dict): The first dataset (e.g., train data).
        data2 (dict): The second dataset (e.g., test data).
        
    Returns:
        Tuple[dict, dict]: The two filtered datasets.
    """
    variants1 = data1["variants"]
    variants2 = data2["variants"]
    
    # Create a unique key (CHROM_POS) for intersection
    key1 = variants1["CHROM"].astype(str) + "_" + variants1["POS"].astype(str)
    key2 = variants2["CHROM"].astype(str) + "_" + variants2["POS"].astype(str)
    
    # Find the set of keys common to both
    common_keys = set(key1).intersection(set(key2))
    
    # Create boolean masks to filter the original dataframes
    mask1 = key1.isin(common_keys)
    mask2 = key2.isin(common_keys)
    
    # Apply the masks to filter both datasets
    data1_filtered = apply_variant_mask(data1, mask1)
    data2_filtered = apply_variant_mask(data2, mask2)
    
    common_count = len(common_keys)
    logger.info(f"Overlapping variants: {common_count}")
    
    return data1_filtered, data2_filtered


def harmonize_and_combine_pred(train: dict, test: dict):
    """
    Harmonize SNPs between train/test datasets based on CHROM+POS.
    Flip alleles if needed, drop ambiguous A/T or C/G SNPs.
    
    Args:
        train, test (dict): load_vcf outputs (with geno, samples, variants).
    
    Returns:
        chrom, pos, ref, alt, gt_combined, samples_combined
    """
    def norm_chr(c): return str(c).replace("chr", "").upper()
    def is_ambiguous(r, a): return {r, a} in [{"A", "T"}, {"C", "G"}]
    def flip_gt(arr): return np.where(arr == 0, 1, np.where(arr == 1, 0, arr))

    tr_vars = train["variants"]
    te_vars = test["variants"]

    # Build keys
    tr_keys = [(norm_chr(c), int(p)) for c, p in zip(tr_vars["CHROM"], tr_vars["POS"])]
    te_keys = [(norm_chr(c), int(p)) for c, p in zip(te_vars["CHROM"], te_vars["POS"])]
    tr_index, te_index = {k: i for i, k in enumerate(tr_keys)}, {k: i for i, k in enumerate(te_keys)}

    common = [k for k in tr_index if k in te_index]
    if not common:
        raise RuntimeError("No common SNPs between train/test.")

    tr_keep, te_keep, flips = [], [], []
    for k in common:
        i, j = tr_index[k], te_index[k]
        r1, a1 = tr_vars.loc[i, "REF"].upper(), tr_vars.loc[i, "ALT"].upper()
        r2, a2 = te_vars.loc[j, "REF"].upper(), te_vars.loc[j, "ALT"].upper()

        if r1 == r2 and a1 == a2:
            tr_keep.append(i); te_keep.append(j); flips.append(False)
        elif r1 == a2 and a1 == r2:
            if is_ambiguous(r1, a1):
                continue
            tr_keep.append(i); te_keep.append(j); flips.append(True)

    tr_keep, te_keep, flips = np.array(tr_keep), np.array(te_keep), np.array(flips)
    chrom = tr_vars.loc[tr_keep, "CHROM"].values
    pos   = tr_vars.loc[tr_keep, "POS"].values
    ref   = tr_vars.loc[tr_keep, "REF"].values
    alt   = tr_vars.loc[tr_keep, "ALT"].values

    gt_tr = train["geno"][:, tr_keep].T  # (variants x samples)
    gt_te = test["geno"][:, te_keep].T

    # Flip alleles in test if needed
    if flips.any():
        vmask = flips
        gt_te[vmask, :] = np.vectorize(lambda g: flip_gt(np.array([g]))[0])(gt_te[vmask, :])

    # Concatenate
    gt_combined = np.concatenate([gt_tr, gt_te], axis=1)  # (variants x all_samples)
    samples_combined = train["samples"] + test["samples"]

    return chrom, pos, ref, alt, gt_combined, samples_combined

def write_vcf_pred(path: str, chroms, poss, refs, alts, gt, samples):
    """
    Write a harmonized SNP VCF from genotype + metadata arrays.
    gt: (variants x samples), values in {0,1,2,-1}
    """
    header = [
        "##fileformat=VCFv4.2",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(len(chroms)):
            calls = []
            for val in gt[i, :]:
                if val == -1:
                    calls.append("./.")
                elif val == 0:
                    calls.append("0/0")
                elif val == 1:
                    calls.append("0/1")
                elif val == 2:
                    calls.append("1/1")
                else:
                    calls.append("./.")
            row = [str(chroms[i]), str(poss[i]), ".", str(refs[i]), str(alts[i]), ".", "PASS", ".", "GT"] + calls
            f.write("\t".join(row) + "\n")

def split_vcf_by_samples_pred(src_vcf, train_samples, test_samples, out_train_vcf, out_test_vcf):
    """
    Split a haplotype VCF into train/test by sample IDs.
    """
    with open(src_vcf) as inp, open(out_train_vcf, "w") as ftr, open(out_test_vcf, "w") as fte:
        for line in inp:
            if line.startswith("##"):
                ftr.write(line); fte.write(line)
            elif line.startswith("#CHROM"):
                header_cols = line.strip().split("\t")
                fixed, samples = header_cols[:9], header_cols[9:]
                idx_tr = [9 + samples.index(s) for s in train_samples if s in samples]
                idx_te = [9 + samples.index(s) for s in test_samples if s in samples]
                ftr.write("\t".join(fixed + train_samples) + "\n")
                fte.write("\t".join(fixed + test_samples) + "\n")
            else:
                parts = line.rstrip().split("\t")
                fixed = parts[:9]
                tr_calls = [parts[i] for i in idx_tr]
                te_calls = [parts[i] for i in idx_te]
                ftr.write("\t".join(fixed + tr_calls) + "\n")
                fte.write("\t".join(fixed + te_calls) + "\n")


def run_rtm_gwas_snpldb(tool_path: str, input_vcf: str, output_prefix: str, settings: dict) -> str:
    """
    Run RTM-GWAS snpldb to generate haplotype VCF and capture its output into logger.
    """
    cmd = [
        tool_path,
        "--vcf", input_vcf,
        "--out", output_prefix,
    ]

    # Only append parameters if they are not None
    if "maf" in settings and settings["maf"] is not None:
        cmd += ["--maf", str(settings["maf"])]
    if "maxlen" in settings and settings["maxlen"] is not None:
        cmd += ["--maxlen", str(settings["maxlen"])]
    if "threads" in settings and settings["threads"] is not None:
        cmd += ["--thread", str(settings["threads"])]

    logger.info(f"Running RTM-GWAS SNPLDB: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output line by line into logger
    for line in process.stdout:
        logger.info(f"[rtm-gwas] {line.strip()}")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"RTM-GWAS failed with exit code {process.returncode}")

    return f"{output_prefix}.vcf"


def haplotype_features_prediction(
    train_geno_path: str,
    test_geno_path: str,
    geno_base: str,
    config: dict
):
    """
    Build haplotype VCFs for train and test using RTM-GWAS,

    Steps:
      1. Load raw train/test VCFs
      2. Intersect common SNPs (CHROM + POS)
      3. Filter to biallelic SNPs
      4. Harmonize & merge SNPs
      5. Run RTM-GWAS haplotype inference
      6. Split haplotype VCF into train/test
      7. Reload haplotype VCFs

    Args:
        train_geno_path (str): Path to training VCF.
        test_geno_path (str): Path to test VCF.
        geno_base (str): Base name for haplotype outputs.
        config (dict): Full pipeline config.

    Returns:
        tuple:
            - train_geno (np.ndarray): Training haplotype genotype matrix.
            - test_geno (np.ndarray): Test haplotype genotype matrix.
            - train_samples (list): Training sample IDs.
            - test_samples (list): Test sample IDs.
            - train_variants (pd.DataFrame): Training haplotype variants.
            - test_variants (pd.DataFrame): Test haplotype variants.
    """

    # Output directory: results_dir/prediction_results/<geno_base>/
    out_dir = os.path.join(config["general"]["results_dir"], "prediction_results", geno_base)
    os.makedirs(out_dir, exist_ok=True)


    logger.info("Loading training VCF...")
    train = load_vcf(train_geno_path)
    logger.info("Loading test VCF...")
    test = load_vcf(test_geno_path)


    train = filter_biallelic_pred(train)
    test = filter_biallelic_pred(test)

    

    logger.info("Identifying common variants based on CHROM and POS.")
    # The return type is a tuple (dict, dict)
    train, test = intersect_variants_by_pos(train, test)


    chrom, pos, ref, alt, gt_combined, samples_combined = harmonize_and_combine_pred(train, test)

    merged_vcf = os.path.join(out_dir, f"{geno_base}_merged_common_harmonized.vcf")


    write_vcf_pred(merged_vcf, chrom, pos, ref, alt, gt_combined, samples_combined)


    hap_prefix = os.path.join(out_dir, f"{geno_base}_hap")
    hap_vcf = run_rtm_gwas_snpldb(
        tool_path=config["feature_view_settings"]["rtm-gwas-snpldbtool"],
        input_vcf=merged_vcf,
        output_prefix=hap_prefix,
        settings={
            "maf": config.get("maf", None),
            "maxlen": config.get("maxlen", None),
            "threads": config.get("threads", 1),
        }
    )


    train_hap_vcf = os.path.join(out_dir, f"{geno_base}_converted_train_hap.vcf")
    test_hap_vcf  = os.path.join(out_dir, f"{geno_base}_converted_test_hap.vcf")

    split_vcf_by_samples_pred(
        hap_vcf,
        train_samples=train["samples"],
        test_samples=test["samples"],
        out_train_vcf=train_hap_vcf,
        out_test_vcf=test_hap_vcf,
    )


    train_data = load_vcf(train_hap_vcf)
    test_data = load_vcf(test_hap_vcf)

    logger.info("Haplotype feature matrices successfully prepared for prediction mode.")

    return (
        train_data["geno"],
        test_data["geno"],
        train_data["samples"],
        test_data["samples"],
        train_data["variants"],
        test_data["variants"],
    )


def haplotype_features(geno_path: str, geno_dir: str, geno_base: str, config: dict):
    """
    Build or load haplotype VCF using RTM-GWAS and return raw genotypes + samples + variants.
    """
    hap_prefix = os.path.join(geno_dir, f"{geno_base}_hap")
    hap_vcf = f"{hap_prefix}.vcf"

    if not os.path.exists(hap_vcf):
        hap_vcf = run_rtm_gwas_snpldb(
            config["feature_view_settings"]["rtm-gwas-snpldbtool"],
            geno_path,
            hap_prefix,
            config["feature_view_settings"],
        )
        logger.info(f"Generated new haplotype VCF: {hap_vcf}")
    else:
        logger.info(f"Using existing haplotype VCF: {hap_vcf}")

    geno_data = load_vcf(hap_vcf)
    return geno_data["geno"], geno_data["samples"], geno_data["variants"]