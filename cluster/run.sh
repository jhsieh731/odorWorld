#!/bin/bash
##SBATCH -p shared
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-03:00
#SBATCH --mem=5G
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=jhsieh@college.harvard.edu


# Load software modules and source conda environment
module load python/3.10.12-fasrc01
source activate pt2.1.0_cuda12.1

# Run script, params: num_odors mix_size dset_type batch_size num_epochs load_model num_samples save_modulo lr
python run.py 500 15 'U' 64 1000 'final_models/U_500_M10.pt' 10000 50 0.000015