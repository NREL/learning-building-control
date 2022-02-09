#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=lbc
#SBATCH --qos=high
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_rlc.py --bsz $BATCH_SIZE --dr $DR_PROGRAM --dry-run $DRY_RUN \
    --results-dir $RESDIR
