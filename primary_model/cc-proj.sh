#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results.out

module load python
source ~/.proj/bin/activate
python main.py 