#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=rbc
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_rbc.py --bsz $1 --dr $2 --results-dir $3

