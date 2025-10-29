# multiGS_P

## A Modular Genomic Selection Pipeline Using Python

**multiGS_P** is a modular and configurable **Genomic Selection (GS)** pipeline designed for plant breeding research.  
It provides a unified framework that seamlessly integrates **classical GS**, **machine learning (ML)**, and **deep learning (DL)** models, enabling comprehensive and reproducible genomic prediction for plant breeding programs.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Citation](#citation)
- [Tutorials](#tutorials)
- [License](#license)

## Overview

multiGS_P provides an end-to-end workflow for **Genomic Selection (GS)**, from data preprocessing to model training, evaluation, and prediction. The pipeline streamlines all stages of the genomic selection process, enabling **reproducible** and **scalable** genomic prediction across diverse datasets and environments. The pipeline supports multiple **genomic marker representations** including SNPs, Haplotypes, and Principal Components along with a broad suite of **statistical, machine learning, and deep learning algorithms**, making it a complete platform for both **cross-validation** and **prediction** workflows.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AAFC-ORDC-Crop-Bioinfomatics/multiGS_P.git
cd multiGS_P
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
## Configuration

All pipeline settings are defined in a single `.ini` configuration file. 

Only a minimal example is shown below.  
For a complete list of available parameters and detailed explanations, refer to the **multiGS_P User Guide (PDF)**.

### Example

```ini
[General]
seed = 42
n_folds = 5
results_dir = /path/to/results

[InputData]
train_geno_path = /path/to/train_genotypes
train_pheno_path = /path/to/train_phenotypes

[Models]
RANDOMFOREST = True
XGBOOST = True
MLP = True

[FeatureView]
feature_view = SNP
```
## Usage

Once the configuration file (`config.ini`) is prepared, the pipeline can be executed directly as a Python module.

### Run the Pipeline

```bash
python -m multiGS_P.main --config config.ini
```

## Citation

If you use **multiGS_P** in your research, please cite it as follows:
> *Frank M. You¹, Chunfang Zheng¹, John Joseph Zagariah Daniel¹, Pingchuan Li¹, Sylvie Cloutier¹, Kenneth Jackle², Megan House², Bunyamin Tar’an² (2025)  multiGS_P: A Modular Genomic Selection Pipeline Using Python.*  

## Tutorials

Comprehensive instructions for configuration, data preparation, model training, and execution are provided in the **multiGS_P User Guide (PDF)**.

The guide will be available in the repository’s `docs/` directory.

## License

This project is licensed under the terms of the **MIT License**.