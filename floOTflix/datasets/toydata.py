import ot
import numpy as np
from typing import Tuple
import torch

from ..constants import device

TDataSet = Tuple[torch.TensorType, torch.TensorType]
THiddenVar = Tuple[torch.TensorType, torch.TensorType, torch.TensorType]


def _generate_dataset(m: int, n: int, k: int, eps: float, train_size: float):
    # create random marginal distributions
    r = 1 + torch.rand(m).to(device)
    c = 1 + torch.rand(n).to(device)

    r /= torch.sum(r)
    c /= torch.sum(c)

    # create random cost matrix
    U_truth = torch.rand(m, k).to(device)
    V_truth = torch.rand(n, k).to(device)
    A_truth = torch.rand(k, k).to(device)

    C_truth = U_truth @ A_truth @ V_truth.t()

    pi_sample = ot.sinkhorn(r, c, C_truth, reg=eps, numItermax=100_000)

    N = m * n
    N_train = int(train_size * N)

    train_index = np.random.choice(N, N_train, replace=False)
    test_index = np.setdiff1d(np.arange(N), train_index)

    train_index = np.unravel_index(train_index, (m, n))
    test_index = np.unravel_index(test_index, (m, n))

    X_train = torch.from_numpy(np.array(train_index).T).to(device)
    y_train = pi_sample[train_index]
    pi_train = pi_sample.clone()
    pi_train[test_index] = np.nan

    X_test = torch.from_numpy(np.array(test_index).T).to(device)
    y_test = pi_sample[test_index]
    pi_test = pi_sample.clone()
    pi_test[train_index] = np.nan

    return (X_train, y_train), (X_test, y_test), (pi_train, pi_test, C_truth, U_truth, V_truth, A_truth)


def get_dataset(m: int, n: int, k: int, eps=1e-1, train_size=.75) -> Tuple[TDataSet, TDataSet, THiddenVar]:
    """Generate toy datasets.

    Args:
        m (int): number of rows
        n (int): number of columns
        k (int): number of features
        eps (float, optional): regularisation parameter. Defaults to 1e-1.
        train_size (float, optional): train/test split. Defaults to .75.

    Returns:
        Tuple[TDataSet, TDataSet, THiddenVar]: train and test dataset and hidden variables.
    """
    return _generate_dataset(m, n, k, eps, train_size)
