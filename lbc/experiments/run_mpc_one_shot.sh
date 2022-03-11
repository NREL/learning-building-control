#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=oracle
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_mpc_one_shot.py --bsz $1 --dr $2 --results-dir $3 \
    --control-variance-penalty $4 --name-ext $4

