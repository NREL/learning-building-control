from collections import defaultdict

import numpy as np

import torch


def _maybe_to_numpy(x, data_type):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return np.array(x).squeeze()


class Rollout:
    """Simple container class for solutions / policy rollouts."""

    def __init__(self, batch_size: int = None, time_index: any = None,
                 **kwargs):

        self.batch_size = batch_size
        self.time_index = time_index
        self.data = defaultdict(list)
        self.data_types = {}

    def update_batched(self, **kwargs) -> None:
        self.add(data_type="batched", **kwargs)

    def update_scalar(self, **kwargs) -> None:
        self.add(data_type="scalar", **kwargs)

    def add(self, data_type: str, **kwargs) -> None:
        """Iterates over keyword arguments and adds numpified values to
        defaultdict.
        """

        assert data_type in ["scalar", "batched"]

        for key, x in kwargs.items():
            self.data_types[key] = data_type
            x = _maybe_to_numpy(x, data_type=data_type)
            self.data[key].append(x)

    def finalize(self):
        # Apply np.stack to the values to create single arrays.
        for key, value in self.data.items():
            self.data[key] = np.stack(value)

    def show_dims(self):
        """Prints the dimensions of all rollout variables."""
        keys = sorted(self.data.keys())
        for key in keys:
            value = self.data[key]
            try:
                print(key, value.shape, self.data_types[key])
            except:
                print(f"{key}: no data")
