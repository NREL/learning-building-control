#!/bin/bash
#SBATCH --account=aumc
#SBATCH --job-name=lbc
#SBATCH --qos=high
#SBATCH --time=4:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

mkdir -p ./results

dry=0
bsz=31
dr=TOU

srun python run_mpc_one_shot.py --bsz $bsz --dr $dr --dry-run $dry
srun python run_rlc.py --bsz $bsz --dr $dr --dry-run $dry
srun python run_rbc.py --bsz $bsz --dr $dr --dry-run $dry --p-flow=1 --p-temp=1
srun python run_mpc.py --bsz $bsz --dr $dr --dry-run $dry --lookahead=4
srun python run_dpc.py --bsz $bsz --dr $dr --dry-run $dry --num-epochs=2
srun python run_cpl.py --bsz $bsz --dr=$dr --dry-run $dry --lookahead 4 --num-epochs 2
srun python run_cpl.py --bsz $bsz --dr=$dr --dry-run $dry --lookahead 4 --num-epochs 2 --use-value-function 0

