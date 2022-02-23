#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=mpc
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_mpc.py --bsz $1 --dr $2 --results-dir $3 \
    --lookahead=$4 

