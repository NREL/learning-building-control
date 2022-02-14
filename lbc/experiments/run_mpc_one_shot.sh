#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=oracle
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_mpc_one_shot.py --bsz $BATCH_SIZE --dr $DR_PROGRAM \
    --results-dir $RESDIR

