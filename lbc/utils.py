from typing import Union
from numbers import Number

import numpy as np

import torch


def to_torch(x: Union[list, np.ndarray],
             batch_size: int = None) -> torch.tensor:
    """Converts a number, list, or array to (optionally batched / broadcasted)
    torch tensor."""

    x = [x] if isinstance(x, Number) else x
    x = torch.from_numpy(np.array(x).astype(np.float32))

    if batch_size is not None:
        return x * torch.ones(batch_size, *x.shape)

    return x
