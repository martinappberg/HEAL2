# HEAL2: Deep Learning Framework for Rare Variant Analysis

HEAL2 (Hierarchial Estimate for Agnostic) is a deep learning-powered pipeline for analyzing rare genetic variants using a graph neural network (GNN) architecture integrated with an attention-readout mechanism coupled with a sparse autoencoder (SAE) for interpretability. It is tailored for binary phenotypic classification of large-scale whole-genome sequencing (WGS) studies and supports preprocessing of ANNOVAR annotated VCF files, model training, evaluation, and gene / feature prioritization.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Script Details](#script-details)

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
- TODO! Environmental file

### Input Data Requirements
- ANNOVAR-annotated VCF files
- Population data file (for population filtering)
- Phenotype file
- Gene-gene interaction (GGI) data

## Pipeline Overview

The pipeline consists of three main stages:
1. Data Preprocessing
2. Population Filtering
3. Model Training & Evaluation (Linear and GNN)

## Data Preprocessing

### ANNOVAR Preprocessing
```bash
python scripts/preprocess/preprocess.py \
    --AF <allele_frequency_threshold> \
    --file <input_annovar_file> \
    --n_samples <number_of_samples> \
    --output <output_directory> \
    [--indels] \
    [--gnomad_ancestry <ancestry>] \
    [--equal_allele_weights] \
    [--multidimensional]
```

Key Options:
- `--AF`: Allele frequency threshold for gnomAD filtering
- `--indels`: Include indel variants
- `--gnomad_ancestry`: Specific gnomAD ancestry for filtering
- `--equal_allele_weights`: Use equal weights for heterozygous/homozygous variants
- `--multidimensional`: Include all pathogenicity scores (sum and max)

### Population Filtering
```bash
python scripts/preprocess/filter_pop.py \
    --pop <population> \
    --file <population_file> \
    --directory <mutation_matrices_directory> \
    [--threshold <population_threshold>]
```

### Model Data Preparation
```bash
python scripts/gnn/preprocess_model.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    --gene_list <path_to_gene_list> \
    --pheno <phenotype_file> \
    [--af <allele_frequency>]
```

## Model Training and Evaluation

### Linear Model
```bash
python scripts/linear/linear.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    [--af <allele_frequency>] \
    [--covariates <covariates_file>] \
    [--logo] \
    [--stratified_kfold] \
    [--output <output_directory>]
```

### GNN Model Training

#### Performance Evaluation
```bash
python scripts/gnn/gnn.py \
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
python scripts/gnn/gnn_attention.py \
    --data_path <path_to_data> \
    --dataset <dataset_name> \
    [--af <allele_frequency>] \
    [--covariates <covariates_file>] \
    [--output <output_directory>]
```

## Script Details

### Preprocessing Scripts

#### preprocess.py
- Primary script for processing ANNOVAR-annotated VCF files
- Handles variant filtering, score calculation, and matrix generation

#### filter_pop.py
- Filters samples based on population ancestry
- Uses population-specific thresholds
- Creates population-filtered mutation matrices

### Model Scripts

#### linear.py
- Implements linear model baseline
- Supports various cross-validation strategies
- Includes feature importance analysis

#### gnn.py
- Implements graph neural network model
- Supports various training configurations
- Includes comprehensive evaluation metrics

#### gnn_attention.py
- Specialized script for attention score analysis and running on the full cohort
- Focuses on model interpretability and gene attention weights

### Common Features Across Models

Both linear and GNN models support:
- Leave-one-group-out (LOGO) validation
- Stratified k-fold cross-validation
- Covariate inclusion
- Multiple evaluation metrics (AUROC, AUPRC)
- Feature importance analysis

### Output Files

The pipeline generates several output files:
- Preprocessed mutation matrices
- Model predictions
- Performance metrics
- Attention scores (GNN)
- Feature importance scores
- Validation results

## Notes

- All scripts support extensive command-line arguments for customization
- Use `--help` with any script for detailed parameter information
- Population filtering is recommended before model training
- GNN preprocessing requires additional gene-gene interaction data
