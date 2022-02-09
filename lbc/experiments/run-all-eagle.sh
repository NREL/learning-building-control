#!/bin/bash

# Exit on failure so you can troubleshoot
set -xe

# Shared settings
export DR_PROGRAM=PC
export DRY_RUN=0
export BATCH_SIZE=31

# Results dir, export so scripts have it
export RESDIR=$PWD/results-$DR_PROGRAM
mkdir -p $RESDIR

# RBC
sbatch --time=10:00 run_rbc.sh

# MPC ONE SHOT
sbatch --time=1:00:00 run_mpc_one_shot.sh

# MPC
sbatch --time=1:00:00 run_mpc.sh 4
sbatch --time=1:00:00 run_mpc.sh 8
sbatch --time=1:00:00 run_mpc.sh 12
sbatch --time=2:00:00 run_mpc.sh 24
sbatch --time=6:00:00 run_mpc.sh 36
sbatch --time=8:00:00 run_mpc.sh 48

# CPL No Learning
sbatch --time=1:00:00 run_cpl.sh 4 0
sbatch --time=1:00:00 run_cpl.sh 8 0
sbatch --time=1:00:00 run_cpl.sh 12 0
sbatch --time=1:00:00 run_cpl.sh 24 0
sbatch --time=1:00:00 run_cpl.sh 36 0
sbatch --time=1:00:00 run_cpl.sh 48 0

# CPL with learning
sbatch --time=2:00:00 run_cpl.sh 4 1
sbatch --time=4:00:00 run_cpl.sh 8 1
sbatch --time=8:00:00 run_cpl.sh 12 1

# DPC
sbatch --time=2:00:00 run_dpc.sh

# RLS
sbatch --time=30:00 run_rlc.sh

