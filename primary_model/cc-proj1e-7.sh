#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --nodes=3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results7.out

module load python
source ~/.proj/bin/activate
python main7.py 
