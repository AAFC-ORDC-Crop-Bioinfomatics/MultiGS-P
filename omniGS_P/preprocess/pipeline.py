# omniGS_P/preprocess/pipeline.py

import os
import pandas as pd
import logging
from omniGS_P.utils.io import read_hdf5, write_hdf5
from omniGS_P.preprocess.loaders import load_vcf, load_pheno
from omniGS_P.preprocess.processors import filter_snps, align_geno_pheno, impute_missing, normalize
from omniGS_P.preprocess.processors import intersect_and_order_snps, impute_and_fit_transform, impute_transform, normalize_and_fit_transform, normalize_transform, qc_filter_fit_transform
from omniGS_P.preprocess.representations.snp import snp_features
from omniGS_P.preprocess.representations.pca import pca_features, plot_pca_variance, pca_features_prediction
from omniGS_P.preprocess.representations.haplotype import haplotype_features, haplotype_features_prediction


logger = logging.getLogger(__name__)


def run_preprocessing_cv(config: dict):
    """
    Preprocessing pipeline for Cross-Validation (CV) mode:
      - Load training genotype (VCF or Haplotype VCF) and phenotype
      - Apply QC filtering, alignment, imputation, normalization
      - Generate SNP / Haplotype / PCA representation (based on feature_view)
    """
    # -------------------------
    # Paths
    # -------------------------
    geno_path = config["input"]["train_geno"]
    pheno_path = config["input"]["train_pheno"]
    feature_view = config["feature_view"].lower()
    maf_thresh = float(config["general"]["maf_thresh"])
    miss_thresh = float(config["general"]["miss_thresh"])

    geno_dir = os.path.dirname(geno_path)
    geno_base = os.path.splitext(os.path.basename(geno_path))[0]
    geno_cache = os.path.join(geno_dir, f"{geno_base}_{feature_view}.h5")

    # -------------------------
    # Genotype 
    # -------------------------
    if feature_view == "snp":
        geno_data = load_vcf(geno_path)
        geno = geno_data["geno"]
        sample_ids = geno_data["samples"]
    
        
        geno = impute_missing(geno, method="mean")
        geno = normalize(geno, method="zscore")
    
        logger.info("Generating SNP representation")
        geno = snp_features(geno)
    
    elif feature_view == "hap":
        logger.info("Building haplotype representation via RTM-GWAS")
        geno, sample_ids, variants = haplotype_features(
            geno_path, geno_dir, geno_base, config
        )
    
        
        geno = impute_missing(geno, method="mean")
        geno = normalize(geno, method="zscore")
    
        logger.info("Generating haplotype features")
    
    elif feature_view == "pc":
        geno_data = load_vcf(geno_path)
        geno = geno_data["geno"]
        sample_ids = geno_data["samples"]
    
        
        geno = impute_missing(geno, method="mean")
        geno = normalize(geno, method="zscore")
    
        logger.info("Generating PC representation")
        geno, explained_variance = pca_features(geno, config["feature_view_settings"])
    
        geno_dir_results = os.path.join(
            config["general"]["results_dir"], "cross_validation_results", geno_base
        )
        os.makedirs(geno_dir_results, exist_ok=True)
    
        # Save variance explained table
        var_path = os.path.join(geno_dir_results, "pca_variance.tsv")
        explained_variance.to_csv(var_path, sep="\t", index=False)
    
        # Save cumulative variance plot
        plot_path = os.path.join(geno_dir_results, "pca_variance.png")
        threshold = config["feature_view_settings"].get("pca_variance_threshold")
        plot_pca_variance(explained_variance, plot_path, threshold=threshold)
    
        if threshold:
            pcs_needed = (
                explained_variance["cumulative_variance"] >= threshold
            ).idxmax() + 1
            logger.info(
                f"PCA variance threshold {threshold:.2f} reached at PC{pcs_needed}"
            )
    
        logger.info(
            f"PCA completed: {geno.shape[1]} PCs retained "
            f"(variance summary saved to {var_path}, plot saved to {plot_path})"
        )
    
    else:
        raise ValueError(f"Unknown feature_view: {feature_view}")
    
    # -------------------------
    # Phenotype
    # -------------------------
    pheno = load_pheno(pheno_path)
    geno, pheno = align_geno_pheno(geno, sample_ids, pheno)
    logger.info(
        f"Phenotype aligned: {pheno.shape[0]} samples x {pheno.shape[1]-1} traits"
    )
    
    return geno, pheno

def run_preprocessing_prediction(config: dict):
    """
    Preprocessing pipeline for Prediction mode:
      - Load training and test genotype (VCF or Haplotype VCF)
      - Intersect common features
      - Apply QC, imputation, normalization
      - Generate feature representation
      - Align with train and test phenotypes
    """
    # -------------------------
    # Paths and Config
    # -------------------------
    train_geno_path = config["input"]["train_geno"]
    test_geno_path = config["input"]["test_geno"]
    train_pheno_path = config["input"]["train_pheno"]
    test_pheno_path = config["input"].get("test_pheno") # Optional test pheno path

    feature_view = config["feature_view"].lower()
    maf_thresh = float(config["general"]["maf_thresh"])
    miss_thresh = float(config["general"]["miss_thresh"])

   
    if feature_view == "snp":
        
        logger.info("Loading training and test VCF files for prediction.")
        train_data = load_vcf(train_geno_path)
        test_data = load_vcf(test_geno_path)
    
        train_geno = train_data["geno"]
        test_geno = test_data["geno"]
        train_samples = train_data["samples"]
        test_samples = test_data["samples"]
        train_variants = train_data["variants"]
        test_variants = test_data["variants"]
        
        # Intersect variants 
        train_geno, test_geno, common_variants = intersect_and_order_snps(
            train_variants, test_variants, train_geno, test_geno
        )
        logger.info(f"Genotypes successfully intersected. Common features: {train_geno.shape[1]}")
    
        
        logger.info("Imputing missing values.")
        train_geno, impute_means = impute_and_fit_transform(train_geno, method="mean")
        test_geno = impute_transform(test_geno, impute_means)

        
        logger.info("Normalizing genotypes.")
        train_geno, norm_mean, norm_std = normalize_and_fit_transform(train_geno)
        test_geno = normalize_transform(test_geno, norm_mean, norm_std)

        
        logger.info("Generating SNP representation.")
        train_geno = snp_features(train_geno)
        test_geno = snp_features(test_geno)

    elif feature_view == "hap":
        logger.info("Building haplotype representation for train and test via RTM-GWAS")
    
        train_geno, test_geno, train_samples, test_samples, train_variants, test_variants = haplotype_features_prediction(
            train_geno_path,
            test_geno_path,
            os.path.splitext(os.path.basename(train_geno_path))[0],  # geno_base
            config
        )
    
        logger.info("Imputing missing values.")
        train_geno, impute_means = impute_and_fit_transform(train_geno, method="mean")
        test_geno = impute_transform(test_geno, impute_means)
    
        
        logger.info("Normalizing haplotype genotypes.")
        train_geno, norm_mean, norm_std = normalize_and_fit_transform(train_geno)
        test_geno = normalize_transform(test_geno, norm_mean, norm_std)
    
        logger.info("Haplotype features generated and aligned for prediction.")
        
    elif feature_view == "pc":
        logger.info("Loading training and test VCF files for PCA prediction.")
        train_data = load_vcf(train_geno_path)
        test_data = load_vcf(test_geno_path)
    
        train_geno = train_data["geno"]
        test_geno = test_data["geno"]
        train_samples = train_data["samples"]
        test_samples = test_data["samples"]
        train_variants = train_data["variants"]
        test_variants = test_data["variants"]
    
        # Intersect SNPs
        train_geno, test_geno, common_variants = intersect_and_order_snps(
            train_variants, test_variants, train_geno, test_geno
        )
        logger.info(f"Genotypes successfully intersected. Common features: {train_geno.shape[1]}")
    

        logger.info("Imputing missing values.")
        train_geno, impute_means = impute_and_fit_transform(train_geno, method="mean")
        test_geno = impute_transform(test_geno, impute_means)
    
        
        logger.info("Normalizing genotypes.")
        train_geno, norm_mean, norm_std = normalize_and_fit_transform(train_geno)
        test_geno = normalize_transform(test_geno, norm_mean, norm_std)
    
        # PCA 
        logger.info("Performing PCA.")
        
    
        train_geno, test_geno, explained_variance = pca_features_prediction(
            train_geno, test_geno, config["feature_view_settings"]
        )
    
        geno_base = os.path.splitext(os.path.basename(train_geno_path))[0]
        geno_dir_results = os.path.join(
            config["general"]["results_dir"], "prediction_results", geno_base
        )
        os.makedirs(geno_dir_results, exist_ok=True)
    
        # Save variance explained table
        var_path = os.path.join(geno_dir_results, "pca_variance.tsv")
        explained_variance.to_csv(var_path, sep="\t", index=False)
    
        # Save cumulative variance plot
        plot_path = os.path.join(geno_dir_results, "pca_variance.png")
        threshold = config["feature_view_settings"].get("pca_variance_threshold")
        plot_pca_variance(explained_variance, plot_path, threshold=threshold)
    
        if threshold:
            pcs_needed = (explained_variance["cumulative_variance"] >= threshold).idxmax() + 1
            logger.info(f"PCA variance threshold {threshold:.2f} reached at PC{pcs_needed}")
    
        logger.info(
            f"PCA completed: {train_geno.shape[1]} PCs retained "
            f"(variance summary saved to {var_path}, plot saved to {plot_path})"
        )

    else:
        logger.error(f"Unknown feature_view: {feature_view}")
        raise ValueError(f"Unknown feature_view: {feature_view}")


    # -------------------------
    # 3. Phenotype Alignment
    # -------------------------
    
    # Training Phenotype Alignment
    train_pheno = load_pheno(train_pheno_path)
    train_geno, train_pheno = align_geno_pheno(train_geno, train_samples, train_pheno)
    logger.info(
        f"Training phenotype aligned: {train_pheno.shape[0]} samples x {train_pheno.shape[1]-1} traits"
    )

    # Test Phenotype Alignment (optional)
    if test_pheno_path and os.path.exists(test_pheno_path):
        test_pheno = load_pheno(test_pheno_path)
        test_geno, test_pheno = align_geno_pheno(test_geno, test_samples, test_pheno)
        logger.info(
            f"Test phenotype aligned: {test_pheno.shape[0]} samples x {test_pheno.shape[1]-1} traits"
        )
    else:
        # Create a placeholder dataframe for test samples
        test_pheno = pd.DataFrame({"SampleID": test_samples})
        logger.warning("Test phenotype not provided or not found. Prediction results will not be evaluated.")

    return train_geno, train_pheno, test_geno, test_pheno