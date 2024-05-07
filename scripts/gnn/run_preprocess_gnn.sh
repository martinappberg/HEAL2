#!/bin/bash
#SBATCH --job-name=preprocess_gnn    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mkjellbe@stanford.edu     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=preprocess_gnn_%j.log   # Standard output and error log

pwd; hostname; date

module load conda
conda activate PRS-Net
python preprocess_gnn.py -i /blue/sai.zhang/mkjellbe.stanford/processed_gnn/pop_filtered/exonic_only_safs_imputed_nfe_raregnomadcommoncontrols_commongnomadrarecases_equalalleleweights_gnn_REVEL_0.05_gnomad_True_indels_0.99EURfiltered.csv -o /blue/sai.zhang/mkjellbe.stanford/gnn_dataEUR0.99