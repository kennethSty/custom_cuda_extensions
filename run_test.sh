#!/bin/bash
#SBATCH --partition exercise-eml
#SBATCH --time 01:00:00
#SBATCH --job-name custom_kernel
#SBATCH --output kernel_test_log
#SBATCH --gres=gpu:1

srun python torch/test.py
