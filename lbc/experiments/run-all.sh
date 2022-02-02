#!/bin/bash
dry=1
bsz=3
dr=TOU

python run_mpc_one_shot.py --bsz $bsz --dr $dr --dry-run $dry
python run_rlc.py --bsz $bsz --dr $dr --dry-run $dry
python run_rbc.py --bsz $bsz --dr $dr --dry-run $dry --p-flow=1 --p-temp=1
python run_mpc.py --bsz $bsz --dr $dr --dry-run $dry --lookahead=4
python run_dpc.py --bsz $bsz --dr $dr --dry-run $dry --num-epochs=2
python run_cpl.py --bsz $bsz --dr=$dr --dry-run $dry --lookahead 4 --num-epochs 2
python run_cpl.py --bsz $bsz --dr=$dr --dry-run $dry --lookahead 4 --num-epochs 2 --use-value-function 0
