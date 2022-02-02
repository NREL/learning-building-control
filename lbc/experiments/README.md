Run and analyze experiments using different control schemes and demand response programs.

Each policy has a `run_{policy}.py` script that has argparser help,

```
    python run_{policy}.py --help
```

Unless you specify the `--dry-run` argument at run time, output files are saved to
the `results` directory.  The `analysis.ipynb` notebook runs a simple analysis against
these output files.

