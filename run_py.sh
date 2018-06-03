#!/bin/sh
#SBATCH --time=60:00:00          # Run time in hh:mm:ss
#SBATCH --mem=32000              # Maximum memory required (in megabytes)
#SBATCH --job-name=self-boosting
#SBATCH --partition=scott
#SBATCH --gres=gpu:1

module load anaconda
source activate boosting-env
python -u $@
