#!/bin/bash

# Exit on failure so you can troubleshoot
set -xe

# Shared settings
DR=PC
BSZ=31

# Results dir, export so scripts have it
OUTDIR=$PWD/results-$DR
mkdir -p $OUTDIR

# RBC
sbatch --time=10:00 run_rbc.sh $BSZ $DR $OUTDIR

# MPC ONE SHOT
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 0
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 0.01
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 0.1
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 1
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 10
sbatch --time=1:00:00 run_mpc_one_shot.sh $BSZ $DR $OUTDIR 100

# MPC
sbatch --time=2:00:00 run_mpc.sh $BSZ $DR $OUTDIR 3
sbatch --time=2:00:00 run_mpc.sh $BSZ $DR $OUTDIR 6
sbatch --time=2:00:00 run_mpc.sh $BSZ $DR $OUTDIR 12
sbatch --time=4:00:00 run_mpc.sh $BSZ $DR $OUTDIR 24
sbatch --time=8:00:00 run_mpc.sh $BSZ $DR $OUTDIR 36
sbatch --time=12:00:00 run_mpc.sh $BSZ $DR $OUTDIR 48

# CPL No Learning
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 3 1 0 -1 1 
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 6 1 0 -1 1 
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 12 1 0 -1 1 
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 24 1 0 -1 1 
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 36 1 0 -1 1 
sbatch --time=1:00:00 run_cpl.sh $BSZ $DR $OUTDIR 48 1 0 -1 1 

# CPL with learning
# # 24 time windows (~60 minutes per windows)
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 3 200 1 0.1 24
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 6 200 1 0.1 24
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 12 200 1 0.1 24
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 24 200 1 0.1 24
sbatch --time=16:00:00 run_cpl.sh 8 $DR $OUTDIR 36 200 1 0.1 24
sbatch --time=23:00:00 run_cpl.sh 8 $DR $OUTDIR 48 200 1 0.1 24

# 48 time windows (~30 minutes per windows)
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 3 200 1 0.1 48
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 6 200 1 0.1 48
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 12 200 1 0.1 48
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 24 200 1 0.1 48
sbatch --time=16:00:00 run_cpl.sh 8 $DR $OUTDIR 36 200 1 0.1 48
sbatch --time=23:00:00 run_cpl.sh 8 $DR $OUTDIR 48 200 1 0.1 48

# 96 time windows (~15 minutes per window)
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 3 200 1 0.1 96
sbatch --time=4:00:00 run_cpl.sh 8 $DR $OUTDIR 6 200 1 0.1 96
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 12 200 1 0.1 96
sbatch --time=8:00:00 run_cpl.sh 8 $DR $OUTDIR 24 200 1 0.1 96
sbatch --time=16:00:00 run_cpl.sh 8 $DR $OUTDIR 36 200 1 0.1 96
sbatch --time=23:00:00 run_cpl.sh 8 $DR $OUTDIR 48 200 1 0.1 96


# DPC
sbatch --time=2:00:00 run_dpc.sh $BSZ $DR $OUTDIR 2000 24 0.001
sbatch --time=2:00:00 run_dpc.sh $BSZ $DR $OUTDIR 2000 48 0.001
sbatch --time=2:00:00 run_dpc.sh $BSZ $DR $OUTDIR 2000 96 0.001

# RLC
sbatch --time=30:00 run_rlc.sh $BSZ $DR $OUTDIR

