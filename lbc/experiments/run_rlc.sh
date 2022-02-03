#!/bin/bash --login
#SBATCH --account=aumc
#SBATCH --job-name=lbc
#SBATCH --qos=high
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

source env.sh

bsz=$1
dr=$2
dry=$3

python run_rlc.py --bsz $bsz --dr $dr --dry-run $dry