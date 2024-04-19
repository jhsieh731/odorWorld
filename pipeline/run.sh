#!/bin/bash
##SBATCH -p shared
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-01:00
#SBATCH --mem=5G
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=jhsieh@college.harvard.edu


# Load software modules and source conda environment
module load python/3.10.12-fasrc01
source activate pt2.1.0_cuda12.1

# Run script, params: num_odors mix_size dset_type batch_size num_epochs load_model save_model
python run.py 500 3 'U' 64 100 'U_500_M2_Rsample-shuffle.pt' 'U_500_M3.pt' 10000