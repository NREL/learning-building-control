import re
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from lbc.rollout import Rollout


LABELS = {
    "zone_temp": [f"zone {i+1}" for i in range(5)],
    "zone_flow": [f"zone {i+1}" for i in range(5)],
    "dischrage_temp": ["discharge temp"]
}


def _batch_avg_2d(x, aggfun):
    """Takes average over last (batch) dimension and applies
    aggregation function on the result."""
    return aggfun(x.mean(-1))


def summarize_rollout(
    rollout: Rollout,
    aggfun: Callable = np.sum
) -> dict:
    """Summarizes the rollout costs, power, and comfort."""

    keys = rollout.data.keys()

    result = {}
    regexes = [r".*cost", r".*power", r"comfort_viol_deg_hr"]
    for r in regexes:
        for key in [k for k in keys if re.match(r, k)]:
            x = _batch_avg_2d(rollout.data[key], aggfun)
            result[key] = x

    return pd.DataFrame(result, index=["value"]).T


def plot_stats(rollout, key, cmap=cm.brg, cols=None, secondary=None,
               ax=None, figsize=(8, 4)):

    if key == "zone_flow":
        data = rollout.data["clipped_action"][:, :, :-1]
    elif key == "discharge_temp":
        data = rollout.data["clipped_action"][:, :, -1]
    else:
        data = rollout.data[key]
    index = rollout.time_index

    mu = np.mean(data, axis=1)
    data_max = np.max(data, axis=1)
    data_min = np.min(data, axis=1)

    if cols is None:
        if np.ndim(mu) > 1:
            cols = LABELS[key] if key in LABELS else range(mu.shape[-1])
        else:
            cols = [key]

    m = pd.DataFrame(mu, columns=cols, index=index)
    y1 = pd.DataFrame(data_min, index=index)
    y2 = pd.DataFrame(data_max, index=index)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    colors = cmap(np.linspace(0, 1, 6))

    for i, col in enumerate(m.columns):
        _ = m[col].plot(ax=ax, c=colors[i], legend=True)
        ax.fill_between(
            x=index, y1=y1[i], y2=y2[i], alpha=0.2, color=colors[i])

    if secondary is not None and secondary in rollout.data.keys():
        _x = pd.DataFrame(
            rollout.data[secondary].mean(-1), columns=[secondary], index=index)
        _x.plot(secondary_y=True, ax=ax, linestyle=":", c="k")

    if key == "zone_temp":
        comfort_min = pd.DataFrame(
            rollout.data["comfort_min"][:, 0], index=index,
            columns=["comfort min"])
        comfort_max = pd.DataFrame(
            rollout.data["comfort_max"][:, 0], index=index,
            columns=["comfort max"])
        comfort_min.plot(ax=ax, linestyle=":", c="k")
        comfort_max.plot(ax=ax, linestyle=":", c="k")

    return fig, ax
