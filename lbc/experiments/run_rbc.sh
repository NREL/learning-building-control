#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=rbc
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_rbc.py --bsz $BATCH_SIZE --dr $DR_PROGRAM \
  --results-dir $RESDIR

