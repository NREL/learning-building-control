Run and analyze experiments using different control schemes and demand response programs.

Each policy has a `run_{policy}.py` script that has argparser help,

```
    python run_{policy}.py --help
```

Common arguments for all policies are:

```
    --dry-run {0,1}
    --bsz {batch_size}
    --dr {TOU,RTP,PC}
```

Setting `--dry-run=1` enables rapid testing: a very short scenario is used, and output
is not saved.  Otherwise, scenarios are 24-hour periods of exogenous data and results are
output to the `results` directory.  The `results.ipynb` notebook runs a simple analysis 
against these output files.

For a quick summary of results for a given DR program,

```
    python results.py TOU # or RTP, PC
```

This will print the loss and cpu time for any relevant output in `results` directory.

Bash scripts are provided to help streamline experiments.  First, modify the 
`shared-env-vars.sh` script to define high-level parameters that will be shared across
experiments: dry run, batch size, and DR program. The bash scripts for each control 
method will import these variables at run time. 

Algorithm-specific configuration should be modified in the relevant scripts.

To run locally,

```
    ./run_{policy}.sh
```

On Eagle, set your SBATCH settings as desired at the top of the run script, then

```
    sbatch run_{policy}.sh
```
