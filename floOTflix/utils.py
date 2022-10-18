import torch
from .constants import device, SEED
import random
import numpy as np


def check_k_last_increasing(l, k):
    """Check if k last elements of l are increasing.

    Args:
        l (List[int]): The list to check.
        k (int): The number of elements to check.

    Returns:
        bool: Whether the k last elements of l are increasing.
    """
    N = len(l)
    return N >= k and all(x1 < x2 for x1, x2 in zip(l[N-k:], l[N+1-k:]))


def tensor_save(obj, name, dataset):
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    torch.save(obj, f'features/data_{dataset}_{name}.pt')


def tensor_load(obj, name, dataset):
    data = torch.load(
        f'features/data_{dataset}_{name}.pt', map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.load_state_dict(data)
    else:
        with torch.no_grad():
            obj[:] = data


def seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)  # To ensure the same data split between experiments.
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def instance_from_map(
    mapping: dict,
    request,
    name: str = "mapping",
    allow_any: bool = True,
):
    """Get an object from a mapping.
    Arguments:
        mapping: Mapping from string keys to classes or instances
        request: A key from the mapping. If allow_any is True, could also be an
            object or a class, to use a custom object.
        name: Name of the mapping used in error messages
        allow_any: If set to True, allows using custom objects.
    Raises:
        ValueError: if the request is invalid
    """

    # Get the class/instance from the request
    if isinstance(request, str):
        if request in mapping:
            instance = mapping[request]
        else:
            raise ValueError(f"{request} doesn't exists for {name}")
    elif allow_any:
        instance = request
    else:
        raise ValueError(f"Object {request} invalid key for {name}")
    return instance
