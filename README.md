# MultiGS-P

## A Genomic Selection Pipeline for Multiple Single Traits Using Diverse Machine Learning and Deep Learning Models and Marker Types

**MultiGS_P** is a comprehensive Python-based genomic selection pipeline for multiple single traits. It integrates **machine learning**, **deep learning**, and **classical statistical models** for genomic prediction. The pipeline supports multiple feature types—including SNPs, haplotypes, and principal components—and provides both **cross-validation (CV)** and **across-population prediction (APP)** modes, enabling robust genomic selection analyses and practical applications in breeding programs..

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input Files](#input-files)
- [Advanced Configuration](#advanced-configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Logging](#logging)
- [Citation](#citation)
- [Tutorials](#tutorials)
- [License](#license)

# Overview

**MultiGS-P** provides an end-to-end workflow for **Genomic Selection (GS)**, from data preprocessing to model training, evaluation, and prediction. The pipeline streamlines all stages of the genomic selection process, enabling **reproducible** and **scalable** genomic prediction across diverse datasets and environments. The pipeline supports multiple **genomic marker representations** including SNPs, Haplotypes, and Principal Components along with a broad suite of **statistical, machine learning, and deep learning algorithms**, making it a complete platform for both **cross-validation** and **across-population prediction** workflows.

# Key Features
### Multiple marker types (Feature Views):
- **SNP marker (SNP)**
  - Raw SNP markers (0,1,2 encoding)
  - Direct use of individual SNP effects
- **Haplotype View (HAP)**
  - Constructed using RTM-GWAS SNPLDB tool
  -	Captures linkage disequilibrium patterns
  -	Reduces dimensionality while preserving genetic information
- **Principal Components (PC)**
  - Dimensionality reduction via PCA
  -	Configurable variance threshold for component selection
  -	Efficient representation of genetic structure
### Diverse Model Support:
  -	**Machine learning:** 
    - Random Forest (RFR): Ensemble of decision trees
    - XGBoost: Gradient boosting implementation
    - LightGBM: Light gradient boosting machine
  -	**Deep learning:** 
 	- CNN: Convolutional Neural Network for spatial patterns
    - MLP: Multi-Layer Perceptron with advanced features 
  -	**Statistical models:** 
    - **ElasticNet:** Linear model with L1 and L2 regularization
	- **LASSO:** L1-regularized linear model
	- **BRR:** Bayesian Ridge Regression

  -	**Ensemble:** 
   	- Stacking: Meta-ensemble combining multiple base models
	- Any models implemented in this pipeline can be stacked.

### Comprehensive Analysis:
 - Phenotype analysis, 
 - visualization, and 
 - statistical reporting

# Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AAFC-ORDC-Crop-Bioinfomatics/MultiGS-P.git
cd MultiGS-P
```
### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate multigs_p
```
### 3. Update PYTHONPATH
```bash
# Update the full path
export PYTHONPATH=<path>/multiGS_P:$PYTHONPATH
```
### 4. Installing *rtm-gwas-snpldb* tool
The rtm-gwas-snpldb tool for haplotype block identification is included in the MultiGs-R package. The latest executable can also be downloaded separately from:
https://github.com/njau-sri/rtm-gwas

# Configuration

All pipeline settings are defined in a single `.ini` configuration file. 

A complete sample configuration file is provided. Only a few sections may need to be modified—see the user guide for details.

### Example

```ini
[General]
seed = 42
n_replicates = 1
n_folds = 5

# Pipeline will create this folder, if did n't exist
results_dir = /path/to/results_dir

[Data]
# Required for either cross-validation (CV) or across-population prediction (APP) mode
vcf_path = /path/to/train_genotype.vcf
phenotype_path = /path/to/train_phenotype.txt

# Required only for APP mode. If CV mode, this section should be commented
test_vcf_path = /path/to/test_genotype.vcf
test_phenotype_path = /path/to/test_phenotype.txt

# For data preprocessing
pheno_normalization = standard
genotype_normalization = standard

# for principal component (PC) marker type only
pca_variance_explained = 0.95  # Use enough components to explain 95 percent variance

[Tool]
# For HAP marker only
rtm-gwas-snpldb_path = /home/frankyou/gspipeline/rtm-gwas/bin/rtm-gwas-snpldb

[FeatureView]
# Marker type options: SNP | PC | HAP
# Default: SNP
feature_view = HAP      

[Models]
# Classical GS
BRR = True

# Linear / Regression-based
ElasticNet = True
LASSO = True

# Tree-based
RandomForest = True
XGBoost = True
LightGBM = True

# Neural Network Based
CNN = True
MLP = True

# Ensemble
Stacking = True

# Hyper-Parameters for all available Models

# ================================
# Linear / Regression-based
# ================================

[Hyperparameters_ElasticNet]
# Reduce regularization for ElasticNet: from, 1 to 0.1->0.01->0.001
alpha = 1.0
l1_ratio = 0.1   # toward ridge: from 0.5 to 0.1-0.3

[Hyperparameters_LASSO]
alpha = 0.01

# ================================
# Tree-based
# ================================

[Hyperparameters_RandomForest]
n_estimators = 100
max_depth = None

[Hyperparameters_XGBoost]
n_estimators = 500
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
random_state = 42

[Hyperparameters_LightGBM]
n_estimators = 500
max_depth = -1
learning_rate = 0.1
num_leaves = 31
subsample = 0.8
colsample_bytree = 0.8
random_state = 42

# ================================
# Neural Networks
# ================================

[Hyperparameters_MLP]
hidden_layers = 1024,512,256
#activation = gelu
activation = relu
norm = layer
residual = true
input_dropout = 0.05
dropout = 0.5

learning_rate = 0.0005
weight_decay = 0.0015
batch_size = 16
epochs = 500
early_stopping_patience = 40
warmup_ratio = 0.1
grad_clip = 1.0
seeds = 3

use_huber = true
huber_delta = 1.0

swa = true
swa_start = 0.7
swa_freq = 1

[Hyperparameters_CNN]
hidden_channels = 128,128,256
kernel_size = 7
pool_size = 2
learning_rate = 0.0005
batch_size = 32
epochs = 500
dropout = 0.5
weight_decay = 0.001
grad_clip = 1.0
early_stopping_patience = 30
warmup_ratio = 0.1
seeds = 3

# ================================
# Ensemble
# ================================
[Hyperparameters_Stacking]
# Any models implemented in this pipeline can be used for stacking  
base_models = BRR, MLP
meta_model = linear
meta_alpha = 1.0

```

# Input Files
## Genotype Data
- VCF format: Standard Variant Call Format with biallelic SNPs
- VCF file can be gziped (*.gz) 
- VCF file must have standard header lines, containing at least one header line, such as
```ini
##fileformat=VCFv4.2
```
## Phenotype Data
- CSV file or tab-delimited text file with samples as rows and traits as columns
- Support multiple traits
- First column should contain sample IDs
- Missing values are automatically imputed with trait medians

# Advanced Configuration
## PCA Configuration
```ini
[Data]
pca_variance_explained = 0.95  # Auto-select components to explain 95% variance
pca_fit_scope = train  # or 'combined' for train+test
```

## Normalization Options
```ini
[Data]
pheno_normalization = standard  # standard, minmax, robust, or none
genotype_normalization = standard
```
## Across-population prediction
You need to set additional data file for **across-population prediction"
```ini
[Data]
test_vcf_path = new_samples.vcf
test_phenotype_path = new_phenotypes.csv  # Optional for evaluation
```

## Model-specific Hyperparameters
Each model supports extensive hyperparameter tuning through the configuration file.


# Usage

Once the configuration file (`config.ini`) is prepared, the pipeline can be executed directly as a Python module.

### Run the Pipeline
You need to first activate the Conda environment that contains all the required libraries.

```bash
conda activate multigs_p
python MultiGS-P_1.0.pyc --config config.ini
```

# Troubleshooting
## Common Issues
1.	**VCF file errors:** Ensure VCF follows standard format with proper headers
2.	**External tool errors:** Verify RTM-GWAS SNPLDB installation and path configuration
3.	**Convergence warnings:** Adjust hyperparameters or normalization methods

# Logging
Detailed logs are saved in the results directory for debugging.

# Citation

If you use **MultiGS-P** in your research, please cite it as follows:
> *You FM, Zheng C, Zagariah Daniel JJ, Li P, Jackle K,  House M, Tar’an T, Cloutier S. Genomic selection for seed yield prediction achieved through versatile pipelines for breeding efficiency in Flax. (In preparation).*  

# Tutorials

Comprehensive instructions for configuration, data preparation, model training, and execution are provided in the **MultiGS-P User Guide (PDF)**.

The guide will be available in the repository’s `docs/` directory.

# License

This project is licensed under the terms of the **MIT License**.
