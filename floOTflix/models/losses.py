import torch
import numpy as np


def rmse_loss(y_pred, y):
    """Compute the RMSE loss.

    Args:
        y_pred (`R^l`): The predicted values of the selected indices of `U^T * V`.
        y (`R^l`): The true values of the selected indices of `U^T * V`.

    Returns:
        float: The RMSE loss.
    """
    return torch.sqrt(torch.mean((y_pred - y)**2))


def rmse_loss_numpy(y_pred, y):
    """Compute the RMSE loss.

    Args:
        y_pred (`R^l`): The predicted values of the selected indices of `U^T * V`.
        y (`R^l`): The true values of the selected indices of `U^T * V`.

    Returns:
        float: The RMSE loss.
    """
    return np.sqrt(np.mean((y_pred - y)**2))

LOSSES = {
    'rmse': rmse_loss,
}