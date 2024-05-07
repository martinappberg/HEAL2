#!/bin/bash
#SBATCH --job-name=gnn    # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mkjellbe@stanford.edu     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=2:00:00               # Time limit hrs:min:sec
#SBATCH --output=gnn_%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
pwd; hostname; date

module load conda
conda activate PRS-Net

max_iterations=100
counter=0

while [ $counter -lt $max_iterations ]
do
    python gnn.py --num_workers 2 --random_state $counter --data_path /blue/sai.zhang/mkjellbe.stanford/gnn_dataEUR0.99 --dataset af_0.05 --af 0.05 --logo --shuffle_controls
    counter=$((counter + 1))
    echo "Iteration $((counter+1)) completed. Starting over..."
done

echo "Reached the maximum of $max_iterations iterations."