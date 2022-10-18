import torch
import torch.nn as nn
import ot
import optuna

from ..constants import device
from ..interfaces import IModel
from .losses import LOSSES, rmse_loss
from ..utils import check_k_last_increasing, instance_from_map

OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}


class CLIDIOTModel(IModel):
    def __init__(self, m, n, eps=10e-1, max_clidiot_iter=100, max_prox_iter=100, prox_lr=10e-2, prox_gamma=0.5, prox_reg=1, eps_num_stability=10e-6, c0=None, alpha0=None, beta0=None, ot_lambda=1e0, verbose=False):
        self.m = m
        self.n = n
        self.eps = eps
        self.max_clidiot_iter = max_clidiot_iter
        self.max_prox_iter = max_prox_iter
        self.prox_lr = prox_lr
        self.prox_gamma = prox_gamma
        self.prox_reg = prox_reg
        self.eps_num_stability = eps_num_stability
        self.verbose = verbose
        self.c0 = c0 if c0 is not None else torch.ones(m, n)
        self.alpha0 = alpha0 if alpha0 is not None else torch.ones(m)
        self.beta0 = beta0 if beta0 is not None else torch.ones(n)
        self.ot_lambda = ot_lambda
        self.pi_inv_coeff = None

    @staticmethod
    def optuna_trial(m, n, X_train, y_train, X_test, y_test, U, V, trial: optuna.Trial):
        eps = trial.suggest_float('eps', 1e-3, 1e1)
        max_clidiot_iter = trial.suggest_int('max_clidiot_iter', 100, 500)
        max_prox_iter = trial.suggest_int('max_prox_iter', 100, 500)
        # max_clidiot_iter = 100
        # max_prox_iter = 100
        prox_lr = trial.suggest_float('prox_lr', 1e-3, 1e1)
        prox_gamma = trial.suggest_float('prox_gamma', 1e-3, 1e1)
        prox_reg = trial.suggest_float('prox_reg', 1e-3, 1e1)
        init = trial.suggest_categorical(
            'c0', ['ones', 'random', '1e-1', '1e-2', 'mean'])
        ot_lambda = trial.suggest_float('ot_lambda', 1e-3, 1e1)
        init = 'mean'

        if init == 'ones':
            c0 = torch.ones(m, n)
            alpha0 = torch.ones(m)
            beta0 = torch.ones(n)
        elif init == 'random':
            c0 = torch.rand(m, n)
            alpha0 = torch.rand(m)
            beta0 = torch.rand(n)
        elif init == '1e-1':
            c0 = torch.ones(m, n) * 1e-1
            alpha0 = torch.ones(m) * 1e-1
            beta0 = torch.ones(n) * 1e-1
        elif init == '1e-2':
            c0 = torch.ones(m, n) * 1e-2
            alpha0 = torch.ones(m) * 1e-2
            beta0 = torch.ones(n) * 1e-2
        elif init == 'mean':
            c0 = torch.ones(m, n) / (m * n)
            alpha0 = torch.ones(m) / (m * n)
            beta0 = torch.ones(n) / (m * n)

        model = CLIDIOTModel(m, n, eps=eps, max_clidiot_iter=max_clidiot_iter, max_prox_iter=max_prox_iter, prox_lr=prox_lr,
                             prox_gamma=prox_gamma, prox_reg=prox_reg, c0=c0, alpha0=alpha0, beta0=beta0, ot_lambda=ot_lambda, verbose=True)
        train_errors, test_errors = model.fit(
            X_train, y_train, X_test, y_test, U, V)
        return train_errors[-1]

    def clidiot(self, pi_sample, prox_fn, logger=None):
        ens = self.eps_num_stability
        eps = self.eps
        max_iter = self.max_clidiot_iter

        mu = pi_sample@torch.ones(self.n)
        nu = pi_sample.t()@torch.ones(self.m)
        c, alpha, beta = self.c0, self.alpha0, self.beta0
        u, v = torch.exp(alpha / eps), torch.exp(beta / eps)

        for epoch in range(max_iter):
            K = torch.exp(-c / eps)
            u = mu / (K@v + ens)
            v = nu / (K.t()@u + ens)
            K = pi_sample / (torch.outer(u, v) + ens)
            c = prox_fn(-eps * torch.log(K))

            # check early stopping
            # if torch.norm(c_next - c) < delta:
            #     progress.console.log('Converged!')
            #     break

            # log
            if logger:
                logger(epoch, c)
        return c

    def set_inverse_transform_fct(self, x_train, y_train, pi_sample):
        x1, x2 = x_train.T
        pi_seen = pi_sample[x1, x2]

        scaling = y_train / torch.clamp(pi_seen, min=1e-9)
        scaling = scaling[torch.nonzero(
            pi_seen > pi_seen.mean() * 0.25)].mean()

        # Experiment : improve scaling coefficient
        # scaling = (scaling * pi_seen.mean()).detach().requires_grad_(True)
        # pi_xy_rescaled = pi_seen / pi_seen.mean()
        # optimizer = torch.optim.Adam([scaling], lr=1e-1)
        # K = 20
        # for epoch in range(10):
        #     optimizer.zero_grad()
        #     y_pred = scaling * pi_xy_rescaled
        #     if epoch >= K//2:
        #         y_pred = torch.clamp(y_pred, min=0.5, max=5)
        #     loss = ((y_pred - y_train)**2).mean()
        #     loss.backward()
        #     optimizer.step()
        # scaling = scaling / pi_seen.mean()

        self.pi_inv_coeff = scaling

    def inverse_transform_pi(self, pi_values):
        y_pred = pi_values * self.pi_inv_coeff
        return y_pred

    def ot(self, c=None, max_iter=1000):
        if c is None:  # takes from the last time the model fits
            c = self.c
        a = torch.ones(self.m) / self.m
        b = torch.ones(self.n) / self.n
        pi_pred = ot.sinkhorn(a, b, c, reg=self.ot_lambda, max_iter=max_iter)
        if abs(pi_pred.sum()-1.) > 1e-6:
            print('[ERRROR] pi_pred.sum()', float(pi_pred.sum()))
            pi_pred = pi_pred + 1e-9
            pi_pred = pi_pred / pi_pred.sum()
        return pi_pred

    def predict(self, X, c=None, pi_pred=None):
        if pi_pred is None:
            pi_pred = self.ot(c=c)
        y_pred = self.inverse_transform_pi(pi_pred)
        return y_pred[X[:, 0], X[:, 1]]

    def fit(self, X_train, y_train, X_test, y_test, U=None, V=None, loss='rmse'):
        # create the sample pi
        pi_train = torch.zeros((self.m, self.n))
        pi_train[X_train[:, 0], X_train[:, 1]] = y_train
        pi_train += self.eps_num_stability
        pi_train /= pi_train.sum()

        pi_test = torch.zeros((self.m, self.n))
        pi_test[X_test[:, 0], X_test[:, 1]] = y_test
        pi_test += self.eps_num_stability
        pi_test /= pi_test.sum()

        assert U is not None and V is not None, 'U and V must be provided for affinity cost'
        U_inv = torch.linalg.pinv(U)
        V_inv = torch.linalg.pinv(V)

        def prox_fn(c):
            A_hat = U_inv @ c @ V_inv.t()
            A = torch.nn.Parameter(A_hat.clone())
            opt = torch.optim.Adam([A], lr=self.prox_lr)
            for i in range(self.max_prox_iter):
                reg = self.prox_reg * 1/2 * (A**2).sum()
                loss = self.prox_gamma * 1/2 * ((A - A_hat)**2).sum() + reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                A.clamp(min=1e-6)
            A = A.detach()
            self.A = A
            return U @ A @ V.t()
            # A = U_inv @ c @ V_inv.t()
            # self.A = A
            # return U @ A @ V.t()

        train_errors = []
        test_errors = []

        def logger(epoch, c):
            pi_pred = self.ot(c=c)
            self.set_inverse_transform_fct(X_train, y_train, pi_pred)
            train_error = self.score(
                X_train, y_train, loss=loss, pi_pred=pi_pred)
            test_error = self.score(X_test, y_test, loss=loss, pi_pred=pi_pred)
            train_errors.append(train_error)
            test_errors.append(test_error)
            if self.verbose and epoch % (self.max_clidiot_iter // 10) == 0:
                print(
                    f'Epoch {epoch}, train error: {train_error}, test error: {test_error}')

        self.c = self.clidiot(pi_train, prox_fn, logger=logger)
        return train_errors, test_errors

    def score(self, X, y, loss='rmse', c=None, pi_pred=None):
        if c is None and pi_pred is None:
            c = self.c
        if loss == 'rmse':
            loss_fn = rmse_loss
        else:
            raise ValueError(f'Unknown loss: {loss}')
        y_pred = self.predict(X, c=c, pi_pred=pi_pred)
        return loss_fn(y, y_pred).item()


class FeaturesCostModel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.u_features, self.v_features = [nn.Sequential(
            nn.Linear(k, k),
            nn.Softplus(),
        ).to(device) for _ in range(2)]
        self.A = nn.Parameter(torch.rand((k, k), device=device))

        # self.cost_fct = nn.Sequential(
        #     nn.Softplus(),
        #     nn.Linear(k, 1),
        #     nn.Softplus(),
        # )
        # self.cost_fct = nn.Softplus()
        self.eps = 1e-6

    def forward(self, U, V):
        U = self.u_features(U)
        V = self.v_features(V)
        C = ((U @ self.A) * V)
        C = C.sum(-1)
        # C = self.cost_fct(C)
        C = C.clamp(min=0.)
        return C.reshape(-1) + self.eps

    def get_real_cost(self, U, V, x, clamp=True):
        x1, x2 = x.T
        C = self(U[x1], V[x2])
        return C


class CLIDIOTGradientModel(IModel):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.correlation_matrix = None
        self.rate_pi_scale = 1
        self.mu, self.nu = None, None

    def _sgd_ot(self, a, b, C, reg=1e-2, reg2=1e2, max_iter=100, optimizer='adam', lr=1e-8):
        a, b = a.view(-1).detach(), b.view(-1).detach()
        C = C.detach()
        optimizer = instance_from_map(OPTIMIZERS, optimizer, "optimizers")

        pi = torch.ones((len(a), len(b))) / (len(a) * len(b))
        pi = pi.clone().to(device).requires_grad_(True)
        optimizer = optimizer([pi], lr=lr)

        for epoch in range(max_iter):
            optimizer.zero_grad()
            loss = (
                (pi * C).sum()
                + reg * (pi * (torch.log(pi + 1e-9) - 1)).sum()
                + reg2 * ((pi.sum(axis=1) - a)**2).sum()
                + reg2 * ((pi.sum(axis=0) - b)**2).sum()
                + reg2 * ((pi.sum() - 1.)**2)
            ).view((1,))

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pi[:] = pi.clamp(min=0, max=1)

        pi = pi / pi.sum()
        return pi.detach()

    def _fpsot_row_loss(self, pi, x, a, nb_cols):
        assert pi.shape == x.shape
        sum_rows = torch.zeros(len(a))
        nb_vals_rows = torch.zeros(len(a))
        sum_rows.index_add_(0, x, pi)
        nb_vals_rows.index_add_(0, x, torch.ones(len(x)))
        nb_vals_rows = torch.maximum(torch.tensor(1.), nb_vals_rows)
        estimated_rows = sum_rows / nb_vals_rows * nb_cols
        return ((a - estimated_rows)**2).sum()

    def _fast_partial_sgd_ot(self, a, b, x, C, reg=1-1, reg2=1e6, max_iter=100, optimizer='adam', lr=1e-8):
        a, b = a.view(-1).detach(), b.view(-1).detach()
        x1, x2 = x.detach().T
        C = C.detach()
        density_factor = len(x) / (len(a) * len(b))
        optimizer = instance_from_map(OPTIMIZERS, optimizer, "optimizers")

        pi = torch.ones(len(x)) / (len(a) * len(b))
        pi = pi.clone().to(device).requires_grad_(True)
        optimizer = optimizer([pi], lr=lr)

        for epoch in range(max_iter):
            optimizer.zero_grad()
            # print('losses',
            #     float((pi * C).sum()),
            #     float((pi * (torch.log(pi + 1e-9) - 1)).sum()),
            #     float(self._fpsot_row_loss(pi, x1, a, len(b))),
            #     float(self._fpsot_row_loss(pi, x2, b, len(a))),
            #     float((pi.sum() - density_factor)**2)
            # )
            loss = (
                (pi * C).sum()
                + reg * (pi * (torch.log(pi + 1e-9) - 1)).sum()
                + reg2 * self._fpsot_row_loss(pi, x1, a, len(b))
                + reg2 * self._fpsot_row_loss(pi, x2, b, len(a))
                + reg2 * ((pi.sum() - density_factor)**2)
            ).view((1,))

            loss.backward()
            optimizer.step()
            # print('sgd loss', float(loss))

            with torch.no_grad():
                pi[:] = pi.clamp(min=0, max=1)

        pi = pi * density_factor / pi.sum()
        pi = pi.detach()

        assert torch.abs(pi.sum()-density_factor) <= 1e-3, f"Sum of {pi.sum()}"
        return pi

    def _ot(self, C, reg=1e-2, max_iter=100):
        a = torch.ones(self.m) / self.m
        b = torch.ones(self.n) / self.n
        print(C.shape, a.shape, b.shape)
        # a, b = self.mu.view(-1), self.nu.view(-1)
        pi_pred = ot.sinkhorn(a, b, C, reg=reg, max_iter=max_iter,
                              # method='sinkhorn_epsilon_scaling'
                              )
        # pi_pred = self._sgd_ot(self.mu, self.nu, C, reg=reg, max_iter=max_iter)

        assert torch.abs(pi_pred.sum()-1) <= 1e-3, f"Sum of {pi_pred.sum()}"
        return pi_pred

    def predict(self, x, correlation_matrix):
        pi_pred = self._ot(correlation_matrix)
        ratings = pi_pred * self.rate_pi_scale
        ratings = torch.clamp(ratings, min=0.5, max=5)
        return ratings[x[:, 0], x[:, 1]]

    def score(self, x, y, loss='rmse', C=None):
        # C = self.correlation_matrix if C is None else C
        loss_fn = instance_from_map(LOSSES, loss, 'loss')
        y_pred = self.predict(x, C)
        print('y_pred', y_pred.sum())
        return loss_fn(y, y_pred).detach().cpu().numpy()

    def _partial_pi_to_ratings(self, x, y_train, partial_pi):
        x = x.T[1]
        x_train = x[:len(y_train)]
        pi_seen = partial_pi[:len(y_train)]
        n = int(x.max())+1

        # Experiment 1: affine transform with matrix inverse
        # s_xi = torch.zeros(n)
        # s_xi.index_add_(0, x_train, pi_seen)
        # s_xi2 = torch.zeros(n)
        # s_xi.index_add_(0, x_train, pi_seen**2)
        # s_ri = torch.zeros(n)
        # s_xi.index_add_(0, x_train, y_train)
        # s_xri = torch.zeros(n)
        # s_xi.index_add_(0, x_train, y_train * pi_seen)

        # denom = s_xi2 - s_xi**2
        # alpha = (s_xri - s_xi * s_ri) / denom
        # beta = (s_xi2 * s_xi - s_xi * s_xri) / denom

        # alpha = torch.nan_to_num(alpha, self.rate_pi_scale)
        # beta = torch.nan_to_num(beta, 0)

        # Experiment 2: linear transform
        # r = (y_train / torch.maximum(pi_seen, pi_seen.mean()/10))
        # print(r, self.rate_pi_scale)
        # r = r.mean()
        # print(r)
        # print(y_train.min(), y_train.max(), y_train.mean())
        # print(pi_seen.min(), pi_seen.max(), pi_seen.mean(), pi_seen.sum())
        # r = self.rate_pi_scale
        # alpha = torch.ones(n) * r
        # beta = torch.zeros(n)
        # ratings = partial_pi * alpha[x] + beta[x]

        # Experiment 3: average of each pi value for same rating
        # with torch.no_grad():
        #     avg_pi_by_rating = torch.zeros(8)
        #     nb_of_rating = torch.zeros(8)
        #     scaled_y = torch.round(y_train*2).int() - 3
        #     assert (scaled_y>=0).all() and (scaled_y<8).all()

        #     avg_pi_by_rating.index_add_(0, scaled_y, pi_seen)
        #     nb_of_rating.index_add_(0, scaled_y, torch.ones(len(scaled_y)))
        #     avg_pi_by_rating = avg_pi_by_rating / nb_of_rating

        #     lowest_rating = y_train.min()
        #     ratings = torch.ones(len(partial_pi)) * lowest_rating
        #     for pi1, pi2 in zip(avg_pi_by_rating, avg_pi_by_rating[1:]):
        #         ratings += torch.relu(partial_pi-pi1) / 2 / (pi2 - pi1)
        #         ratings -= torch.relu(partial_pi-pi2) / 2 / (pi2 - pi1)

        # Experiment 4: find best coefficient with linear transform
        # scaling = torch.tensor(3.).requires_grad_(True)
        # pi_xy_rescaled = pi_seen / pi_seen.mean()
        # optimizer = torch.optim.Adam([scaling], lr=1e-1)
        # K = 20
        # for epoch in range(10):
        #     optimizer.zero_grad()
        #     y_pred = scaling * pi_xy_rescaled
        #     if epoch >= K//2:
        #         y_pred = torch.clamp(y_pred, min=0.5, max=5)
        #     loss = ((y_pred - y_train)**2).mean()
        #     loss.backward()
        #     optimizer.step()
        # ratings = partial_pi / pi_seen.mean() * scaling

        # Experiment 5: linear transform, only taking higher coefficients
        scaling = y_train / torch.clamp(pi_seen, min=1e-9)
        scaling = scaling[torch.nonzero(
            pi_seen > pi_seen.mean() * 0.25)].mean()
        ratings = partial_pi * scaling

        # Simple solution: scale back by self.rate_pi_scale
        # ratings = partial_pi * self.rate_pi_scale

        ratings = torch.clamp(ratings, min=0.5, max=5)
        return ratings

    def _partial_score(self, partial_ratings, y, loss='rmse'):
        assert partial_ratings.shape == y.shape
        loss_fn = instance_from_map(LOSSES, loss, 'loss')
        return loss_fn(y, partial_ratings).detach().cpu().numpy()

    def partial_train_test_score(self, U, V, cost_model, X_train, Y_train, X_test, Y_test, loss='rmse'):
        x_full = torch.concat([X_train, X_test])
        partial_C = cost_model.get_real_cost(U, V, x_full)
        partial_pi = self._fast_partial_sgd_ot(
            self.mu, self.nu, x_full, partial_C)
        partial_ratings = self._partial_pi_to_ratings(
            x_full, Y_train, partial_pi)
        self.partial_pi = partial_pi  # TODO: remove

        train_error = self._partial_score(
            partial_ratings[:len(X_train)], Y_train, loss=loss)
        test_error = self._partial_score(
            partial_ratings[len(X_train):], Y_test, loss=loss)
        return train_error.item(), test_error.item()

    @staticmethod
    def _get_sparse_marginal(n, m, x, pi):
        mu = torch.zeros(n, 1, device=device)
        nb_items = torch.zeros(n, 1, dtype=int, device=device)

        mu.index_add_(0, x, pi)
        nb_items.index_add_(0, x, torch.ones(len(x), dtype=int, device=device))
        mu = mu / torch.maximum(nb_items, torch.tensor(1.)) * m
        return mu / mu.sum()

    def clidiot_train(self, x_train, pi_train, U, V, eps, max_iter, delta, optimizer, lr, logger=None):
        (m, k), (n, _) = U.shape, V.shape
        x1, x2 = x_train.T

        self.mu = CLIDIOTGradientModel._get_sparse_marginal(m, n, x1, pi_train)
        self.nu = CLIDIOTGradientModel._get_sparse_marginal(n, m, x2, pi_train)

        alpha = torch.rand((m, 1), requires_grad=True, device=device)
        beta = torch.rand((n, 1), requires_grad=True, device=device)
        cost_model = FeaturesCostModel(k)

        optimizer = instance_from_map(OPTIMIZERS, optimizer, "optimizers")
        optimizer = optimizer(
            [alpha, beta] + list(cost_model.parameters()), lr=lr)

        # with Progress(transient=True) as progress:
        # task = progress.add_task("Training CLIDIOT...", total=max_iter)
        for epoch in range(max_iter):
            optimizer.zero_grad()
            C = cost_model(U[x1], V[x2])

            loss = (
                eps * torch.exp(
                    alpha[x1].view(-1) + beta[x2].view(-1) - C.view(-1) / eps
                ).mean()
                - (alpha.view(-1) * self.mu.view(-1) / m).mean()
                - (beta.view(-1) * self.nu.view(-1) / n).mean()
                + (C.view(-1) * pi_train.view(-1)).mean()
            ).view((1,))

            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     C[:] = C.clamp(min=0)

            # check early stopping
            # if torch.norm(c_next - c) < delta:
            #     progress.console.log('Converged!')
            #     break

            # if epoch % 10 == 0:
            if logger:
                # logger(progress, epoch, C)
                logger(None, epoch, cost_model, float(loss))

            # progress.advance(task)
        return C

    def fit(self, X_train, y_train, X_test, y_test, U, V, loss='rmse', eps=1e-2, max_iter=1000, delta=1e-2, optimizer='adam', lr=1e-1):
        (m, _), (n, _) = U.shape, V.shape
        train_errors, test_errors = [], []

        def logger(progress, epoch, cost_model, train_loss):
            if epoch % 10 == 0:
                train_error, test_error = self.partial_train_test_score(
                    U, V, cost_model, X_train, y_train, X_test, y_test, loss=loss
                )
                # r_C = (U @ cost_model.A @ V.T).clamp(min=1e-6)
                # train_error = self.score(X_train, y_train, C=r_C, loss=loss)
                # test_error = self.score(X_test, y_test, C=r_C, loss=loss)
                train_errors.append(train_error)
                test_errors.append(test_error)
                # progress.console.log(
                #     f'Epoch {epoch}, train error: {train_error}, test error: {test_error}')
                print(
                    f'Epoch {epoch}, train loss {train_loss}, train error: {train_error}, test error: {test_error}')

        self.rate_pi_scale = y_train.sum() * (m*n) / len(y_train)
        pi_train = y_train / self.rate_pi_scale
        self.pi_train = pi_train  # TODO: remove

        self.correlation_matrix = self.clidiot_train(
            X_train, pi_train, U, V,
            eps=eps,
            max_iter=max_iter,
            delta=delta,
            optimizer=optimizer,
            lr=lr,
            logger=logger
        )
        return train_errors, test_errors
