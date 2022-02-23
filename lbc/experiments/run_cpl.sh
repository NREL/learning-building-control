#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=cpl
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_cpl.py --bsz $1 --dr=$2 --results-dir $3 \
    --lookahead $4 --num-epochs $5 --use-value-function $6 --lr=$7 \
    --name-ext $7 --num-time-windows $8
