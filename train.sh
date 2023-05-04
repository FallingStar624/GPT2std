#!/bin/bash

#SBATCH --job-name=INC1
#SBATCH --exclude=b00
#SBATCH --nodes=1 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=100G
#SBATCH --partition=3090

source /home/n1/${USER}/.bashrc
source /home/n1/${USER}/anaconda/bin/activate

conda activate torch

srun python ./pretrain_small.py $@

