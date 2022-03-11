#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=lbc
#SBATCH --qos=high
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_rlc.py --bsz $1 --dr $2 --results-dir $3
