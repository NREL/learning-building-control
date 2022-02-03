#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=lbc
#SBATCH --qos=high
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh
source shared-env-vars.sh

python run_mpc.py --bsz $BATCH_SIZE --dr $DR_PROGRAM --dry-run $DRY_RUN --lookahead=2

