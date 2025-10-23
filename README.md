# OmniGS-P

## A Modular Genomic Selection Pipeline Using Python

**OmniGS-P** is a modular and configurable **Genomic Selection (GS)** pipeline designed for plant breeding research.  
It provides a unified framework that seamlessly integrates **classical GS**, **machine learning (ML)**, and **deep learning (DL)** models, enabling comprehensive and reproducible genomic prediction for plant breeding programs.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Citation](#citation)
- [Documentation](#documentation)
- [License](#license)

## Overview

OmniGS-P provides an end-to-end workflow for **Genomic Selection (GS)**, from data preprocessing to model training, evaluation, and prediction. The pipeline streamlines all stages of the genomic selection process, enabling **reproducible** and **scalable** genomic prediction across diverse datasets and environments. The pipeline supports multiple **genomic marker representations** — including SNPs, Haplotypes, and Principal Components — along with a broad suite of **statistical, machine learning, and deep learning algorithms**, making it a complete platform for both **cross-validation** and **prediction** workflows.

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:AAFC-ORDC-Crop-Bioinfomatics/OmniGS-P.git
cd OmniGS-P
```
### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate omnigs_env
```
### 3. Update PYTHONPATH
```bash
export PYTHONPATH=<path>/omniGS-P:$PYTHONPATH
```
## Configuration

All pipeline settings are defined in a single `.ini` configuration file. 

Only a minimal example is shown below.  
For a complete list of available parameters and detailed explanations, refer to the **OmniGS-P User Guide (PDF)**.

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
python -m omniGS.main --config config.ini
```

## Citation

If you use **OmniGS-P** in your research, please cite it as follows:
> *Frank M. You¹, Chunfang Zheng¹, Sylvie Cloutier¹, Pingchuan Li¹, John Joseph Zagariah Daniel¹, Kenneth Jackle², Megan House², Bunyamin Tar’an² (2025)  OmniGS-P: A Modular Genomic Selection Pipeline Using Python.*  

## Documentation

Comprehensive instructions for configuration, data preparation, model training, and execution are provided in the **OmniGS-P User Guide (PDF)**.

The guide will be available in the repository’s `docs/` directory.

## License

This project is licensed under the terms of the **MIT License**.