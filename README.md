# HEAL2: Deep Learning Framework for Rare Variant Analysis

HEAL2 (Hierarchial Estimate for Agnostic Learning) is a deep learning-powered pipeline for analyzing rare genetic variants using a graph neural network (GNN) architecture integrated with an attention-readout mechanism coupled with a sparse autoencoder (SAE) for interpretability. It is tailored for binary phenotypic classification of large-scale whole-genome sequencing (WGS) studies and supports model training, evaluation, and gene / feature prioritization.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Script Details](#script-details)
- [Notes](#notes)

## Prerequisites

### Required Dependencies
```bash
python3
pytorch
dgl
pandas
numpy
scikit-learn
```

### Installation
1. Clone the repository
2. Create conda environment using the provided file:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate heal2
```

### Input Data Requirements
- Processed mutational burden per sample and gene
- Population data file (for population filtering)
- Phenotype file
- Gene-gene interaction (GGI) data

## Pipeline Overview

The pipeline consists of two main components:
1. Linear Model (HEAL)
2. Graph Neural Network Model (HEAL2)

## Model Training and Evaluation

### Linear Model (HEAL)
```bash
python scripts/HEAL.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    [--af <allele_frequency>] \
    [--covariates <covariates_file>] \
    [--logo] \
    [--stratified_kfold] \
    [--output <output_directory>]
```

### HEAL2 Model Training

#### Performance Evaluation
```bash
python scripts/HEAL2.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    [--af <allele_frequency>] \
    [--covariates <covariates_file>] \
    [--logo] \
    [--stratified_kfold] \
    [--output <output_directory>]
```

#### Attention Score Analysis
```bash
python scripts/HEAL2_attention.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    [--af <allele_frequency>] \
    [--covariates <covariates_file>] \
    [--output <output_directory>]
```

## Script Details

### Model Scripts

#### HEAL.py
- Implements linear model baseline
- Supports various cross-validation strategies
- Includes feature importance analysis

#### HEAL2.py
- Implements graph neural network model
- Supports various training configurations
- Includes comprehensive evaluation metrics

#### HEAL2_attention.py
- Specialized script for attention score analysis and running on the full cohort
- Focuses on model interpretability and gene attention weights

### Common Features Across Models

Both HEAL and HEAL2 models support:
- Leave-one-group-out (LOGO) validation
- Stratified k-fold cross-validation
- Covariate inclusion
- Multiple evaluation metrics (AUROC, AUPRC)
- Feature importance analysis

### Output Files

The pipeline generates several output files:
- Model predictions
- Performance metrics
- Attention scores (HEAL2)
- Feature importance scores
- Validation results

## Notes

- All scripts support extensive command-line arguments for customization
- Use `--help` with any script for detailed parameter information
- GNN analysis requires additional gene-gene interaction data
