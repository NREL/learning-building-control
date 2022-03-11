#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=rbc
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

# For TOU and RTP
# python run_rbc.py --bsz $1 --dr $2 --results-dir $3 \
#     --p-flow 1.0 --p-temp 0.8

# For PC
python run_rbc.py --bsz $1 --dr $2 --results-dir $3 \
    --p-flow 0.1333  --p-temp 1.0
