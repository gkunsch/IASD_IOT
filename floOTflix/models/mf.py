import torch
import torch.nn as nn
from rich.progress import Progress

from ..constants import device
from ..interfaces import IModel
from .losses import rmse_loss
from ..utils import check_k_last_increasing, tensor_load


class MFModel(IModel):
    def __init__(self, m, n, k):
        '''MF: Matrix Factorization.

        Args:
            m (int): Number of rows.
            n (int): Number of columns.
            k (int): Number of latent dimensions.
        '''
        super().__init__()
        self.U = nn.Parameter(
            torch.rand((m, k), requires_grad=True, device=device))
        self.V = nn.Parameter(
            torch.rand((n, k), requires_grad=True, device=device))

    def parameters(self):
        return [self.U, self.V]

    def load(self, dataset):
        tensor_load(self.U, 'users_matrix', dataset)
        tensor_load(self.V, 'movies_matrix', dataset)

    def get_features(self, x, y):
        U = self.U.detach()
        V = self.V.detach()
        return U, V

    def predict(self, X):
        '''predict the `l` given indices.

        Args:
            X (`R^{l*2}`): Couple of indices `(i, j)` of `U x V` to predict.

        Returns:
            `R^l`: The predicted values of the selected indices of `U^T * V`.
        '''
        U_indices, V_indices = X[:, 0], X[:, 1]
        return (self.U[U_indices] * self.V[V_indices]).sum(axis=1)

    def _fit_epoch(self, X_train, y_train, l2, loss_fn, opt):
        train_errors = []
        for X_train_batch, y_train_batch in zip(X_train, y_train):
            # predict
            y_pred_train = self.predict(X_train_batch)

            # compute losses
            train_loss = loss_fn(y_pred_train, y_train_batch)
            train_errors.append(train_loss.detach().cpu().numpy())

            # compute regularized loss
            reg_loss = l2 * (self.U**2).mean() + (self.V**2).mean()
            train_loss_with_reg = train_loss + reg_loss

            # backprop
            opt.zero_grad()
            train_loss_with_reg.backward()
            opt.step()
        return sum(train_errors)/len(train_errors)

    def fit(self, X_train, y_train, X_test, y_test, optimizer='sgd', n_epochs=10, early_stopping=True, early_stopping_k=3, l2=1, lr=1e-2, loss='rmse', batch_size=2**12):
        '''Fit the model to the data.
        `l = len(X_train) = len(y_train)`.

        Args:
            X_train (`R^{l*2}`): Couple of indices `(i, j)` of `U x V`.
            y_train (`R^l`): The true values of the selected indices of `(i, j)`.
            X_test (`R^{l*2}`): Couple of indices `(i, j)` of U x V.
            y_test (`R^l`): The true values of the selected indices of `(i, j)`.
            optimizer ('sgd' | 'adam' | 'asgd', optional): The optimizer to use. Defaults to `'sgd'`.
            n_epochs (int, optional): The number of n_epochs to perform. Defaults to `10`.
            early_stopping (bool, optional): Whether to stop the training if the loss start increasing. Defaults to `True`.
            early_stopping_k (int, optional): The number of n_epochs to wait before stopping the training if the loss start increasing. Defaults to `3`.
            l2 (float, optional): The l2 regularization parameter. Defaults to `1`.
            lr (float, optional): The learning rate. Defaults to `1e-2`.
            loss ('rmse', optional): The loss to use. Defaults to `'rmse'`.
            batch_size (int, optional): The batch size. Defaults to `2**12`.

        Returns:
            Tuple[List[float], List[float]]: The train and test losses.
        '''
        if optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'asgd':
            opt = torch.optim.ASGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer}')

        if loss == 'rmse':
            loss_fn = rmse_loss
        else:
            raise ValueError(f'Unknown loss: {loss}')

        if batch_size is None:
            X_train = [X_train]
            y_train = [y_train]
        else:
            X_train = torch.split(X_train, batch_size)
            y_train = torch.split(y_train, batch_size)

        train_errors, test_errors = [], []
        with Progress(transient=True) as progress:
            task = progress.add_task("Training MF...", total=n_epochs)
            for epoch in range(n_epochs):
                train_error = self._fit_epoch(X_train, y_train,
                                              l2=l2, loss_fn=loss_fn, opt=opt)
                test_error = self.score(X_test, y_test)

                train_errors.append(train_error)
                test_errors.append(test_error)

                # check early stopping
                if early_stopping and check_k_last_increasing(test_errors, k=early_stopping_k):
                    progress.console.log('Early stopping')
                    break

                # log
                if epoch % (n_epochs // 10) == 0:
                    progress.console.log(
                        f'Epoch {epoch}, train error: {train_error}, test error: {test_error}')
                progress.advance(task)

        return train_errors, test_errors

    def score(self, X, y):
        '''Compute the score of the model on the data.

        Args:
            X (`R^{l*2}`): Couple of indices `(i, j)` of `U x V`.
            y (`R^l`): The true values of the selected indices of `U^T * V`.

        Returns:
            float: The score of the model on the data.
        '''
        with torch.no_grad():
            y_pred = self.predict(X)
            return float(rmse_loss(y_pred, y).cpu())
