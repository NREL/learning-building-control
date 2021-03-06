import glob
import logging
import os
import pickle

import pandas as pd
import numpy as np


logging.basicConfig(level="INFO")
logger = logging.getLogger(__file__)


def main(dr, results_dir=None):

    results_dir = results_dir if results_dir is not None else f"results-{dr}"
    
    files = glob.glob(os.path.join(results_dir, f"*-{dr}*.p"))
    assert len(files) > 0, "no files found!"

    files = sorted(files)
    logger.info(f" Found {len(files)} files: {files}")

    data = {}
    for f in files:
        with open(f, "rb") as x:
            _data = pickle.load(x)
            key = os.path.basename(f)
            result = {
                key: {
                    "test_loss": _data["loss"],
                    "cpu_time": _data["cpu_time"]
                }
            }
            data.update(result)
            logger.info(result)

    df = pd.DataFrame(data).T
    df = df.sort_values("test_loss")
    print(df)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dr",
        type=str,
        help="DR program",
        choices=["TOU", "RTP", "PC"])
    parser.add_argument(
        "--results-dir",
        type=str,
        help="directory containing result pickles",
        default=None
    )
    a = parser.parse_args()

    _ = main(a.dr, a.results_dir)

