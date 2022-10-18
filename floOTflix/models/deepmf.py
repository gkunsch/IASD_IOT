import torch
import torch.nn as nn

from ..constants import device
from ..interfaces import IModel
from .losses import rmse_loss
from ..utils import tensor_load

class DeepMatrixtModel(nn.Module, IModel):
    def __init__(self, nb_users, nb_movies, k):
        super().__init__()

        self.max_rating = 5
        self.n_features = k
        self.nb_users = nb_users
        self.nb_movies = nb_movies

        self.movies_features = nn.Parameter(torch.randn((self.nb_movies, k), device=device))
        self.movies_ratings_func = nn.Sequential(
            nn.Linear(1, k),
            nn.Sigmoid(),
        ).to(device)
        self.combined_func = nn.Sequential(
            nn.Linear(k, k),
            nn.Sigmoid(),
        ).to(device)
        self.result_func = nn.Linear(1, 1).to(device)
        self.to(device)
    
    def load(self, dataset):
        tensor_load(self, 'nn_model', dataset)
    
    def get_features(self, x, y):
        with torch.no_grad():
            V = self.movies_features.detach()
            print('x', x.shape)
            U = self.user_features(torch.arange(0, x[:,0].max()+1), (x, y))
            return U, V
    
    def user_features(self, x_users, rating_data, ignore_movies=None):
        data_x, data_ratings = rating_data
        data_users, data_movies = data_x.T

        users_that_are_here = torch.zeros(self.nb_users, dtype=int, device=device)
        users_that_are_here.index_add_(0, x_users, torch.ones(len(x_users), dtype=int, device=device))
        data_train_used = users_that_are_here[data_users].nonzero().view(-1)
        data_users = data_users[data_train_used]
        data_movies = data_movies[data_train_used]
        data_ratings = data_ratings[data_train_used]

        ratings = self.movies_ratings_func(data_ratings.unsqueeze(-1) / self.max_rating)
        rating_by_movie = self.combined_func(self.movies_features[data_movies] * ratings)

        # Average of rating_by_movie for each user
        rating_by_user = torch.zeros((self.nb_users, self.n_features), device=device)
        nb_ratings = torch.zeros(self.nb_users, device=device)
        rating_by_user.index_add_(0, data_users, rating_by_movie)
        nb_ratings.index_add_(0, data_users, torch.ones(len(data_users), device=device))

        if ignore_movies is not None:
            rating_by_user[x_users] = rating_by_user[x_users] - rating_by_movie[ignore_movies]
            nb_ratings[x_users] = torch.maximum(torch.tensor(1, device=device), nb_ratings[x_users]-1)

        user_features = rating_by_user[x_users] / nb_ratings[x_users].unsqueeze(1)
        
        return user_features

    def forward(self, eval_xs, rating_data, ignore_if_seen=True):
        x_users, x_movies = eval_xs.T
        ignore_movies = x_movies if ignore_if_seen else None
        movie_features = self.movies_features[x_movies]
        user_features = self.user_features(x_users, rating_data, ignore_movies=ignore_movies)
        
        rating = (movie_features * user_features).mean(-1, keepdim=True)
        rating = self.result_func(rating) * self.max_rating
        return rating.view(-1)
    
    def predict(self, X):
        return self(X)

    def score(self, train_ratings, test_ratings):
        with torch.no_grad():
            eval_x, y_ratings = test_ratings
            predicted_ratings = self(eval_x, train_ratings)
            loss = float(rmse_loss(predicted_ratings, y_ratings).cpu())
        return loss

    def fit(
            self,
            train_ratings,
            test_ratings,
            opti,
            steps=10,
            verbose=False,
            # l2=1,
            batch_size=None,
            lr=1e-2,
            test_inverval=50,
        ):
        train_x, train_y = train_ratings
        batch_size = batch_size or len(train_y)

        vars = self.parameters()
        if opti == 'sgd':
            optimizer = torch.optim.SGD(vars, lr=lr)
        elif opti == 'adam':
            optimizer = torch.optim.Adam(vars, lr=lr)
        else:
            assert False, f"Optimizer {opti} doesn't exists"
        
        train_errors, test_errors = [], []
        for i_step in range(steps):
            if verbose:
                print(f"=========== Step {i_step}")
            indexes = torch.randperm(len(train_y))
            data_x, data_y = train_x[indexes], train_y[indexes]
            data_x, data_y = data_x.split(batch_size), data_y.split(batch_size)

            last_losses = []
            for i_batch, (batch_x, batch_y) in enumerate(zip(data_x, data_y)):
                optimizer.zero_grad()
                predicted = self(batch_x, train_ratings, ignore_if_seen=True)
                loss = rmse_loss(predicted, batch_y)
                loss.backward()
                optimizer.step()
                last_losses.append(loss.detach().cpu().numpy())

                if i_batch % test_inverval == 0:
                    avg_loss, last_losses = sum(last_losses)/len(last_losses), []
                    test_loss = self.score(train_ratings, test_ratings)
                    train_errors.append(avg_loss)
                    test_errors.append(test_loss)
                    if verbose:
                        print(f"Batch {i_batch}/{len(data_x)}, avg loss {avg_loss:.4f}, test loss {test_loss:.4f}")
        if steps:
            print(f'After {steps} steps with {opti}, final loss {train_errors[-1]:.4f} (test loss {test_errors[-1]:.4f})')
        
        return train_errors, test_errors