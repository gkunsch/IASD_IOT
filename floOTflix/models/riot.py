# from https://github.com/ruilin-li/Learning-to-Match-via-Inverse-Optimal-Transport

import torch
import scipy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from rich.progress import Progress
import ot

from ..interfaces import IModel
from .losses import rmse_loss_numpy
from ._riot import Matcher, model_parameters, train_parameters, rot


class RIOTModel:
    def __init__(self, m, n, U, V) -> None:
        self.U = U.T
        self.V = V.T
        self.p = U.shape[1]
        self.q = V.shape[1]
        self.m = m
        self.n = n
        self.C = None

    def compute_marginals(self, X, y):
        """Compute the marginals of the optimal transport plan.

        Args:
            X (`R^{l*2}`): Couple of indices `(i, j)` of `U x V` to predict.
            y (`R^l`): The true values of the selected indices of `(i, j)`.

        Returns:
            Tuple[List[Float], List[Float]]: Marginals (`P(X_i), P(X_j)`).
        """
        p_i = np.zeros((self.m,)) + 10e-6  # avoid division by zero
        p_j = np.zeros((self.n,)) + 10e-6  # avoid division by zero

        for i in range(self.m):
            p_i[i] = np.sum(y[X[:, 0] == i])
        for i in range(self.n):
            p_j[i] = np.sum(y[X[:, 1] == i])

        p_i /= np.sum(p_i)
        p_j /= np.sum(p_j)
        return p_i, p_j

    def predict(self, X, reg=.1):
        N = len(X)
        y = np.ones(N)  # same density for every point
        p_i, p_j = self.compute_marginals(X, y)
        pred_pi = ot.sinkhorn(p_i, p_j, self.C, reg=reg)
        pred_pi = pred_pi/pred_pi.sum() * N  # normalize
        return pred_pi[X[:, 0], X[:, 1]], pred_pi

    def fit(self, X_train, y_train):
        pi_sample = np.zeros((self.m, self.n)) + \
            10e-6  # avoid division by zero
        pi_sample[X_train[:, 0], X_train[:, 1]] = y_train
        pi_sample /= np.sum(pi_sample)

        model = Matcher(pi_sample, self.U, self.V, r=5)
        lam = 1
        model_param = model_parameters(A0=np.eye(self.p, self.q),
                                       gamma=0.2,
                                       const=1,
                                       degree=2,
                                       lam=lam,
                                       lambda_mu=1,
                                       lambda_nu=1,
                                       delta=0.005)
        train_param = train_parameters(max_outer_iteration=20,
                                       max_inner_iteration=10,
                                       learning_rate=0.01)
        model.riot(model_param, train_param)
        self.C = model.C

    def score(self, X, y):
        y_pred, _ = self.predict(X)
        return rmse_loss_numpy(y_pred, y.cpu().numpy())
