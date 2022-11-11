#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python main.py 