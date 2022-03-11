#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=dpc
#SBATCH --qos=high
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

set -x

source env.sh

python run_dpc.py --bsz $1 --dr $2 --results-dir $3 \
    --num-epochs $4  --num-time-windows $5 \
    --lr $6

