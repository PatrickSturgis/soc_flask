#!/bin/bash
#SBATCH --job-name=soc_test
#SBATCH --output=test_soc_%j.out
#SBATCH --error=test_soc_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/miniconda3/bin/activate soc_env
cd /data/spack/users/sturgis/winston_transfer
python test_classify_single.py