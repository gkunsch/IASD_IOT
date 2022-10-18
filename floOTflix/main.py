import random

from rich.table import Table
import numpy as np
import torch

from .loader import get_dataset
from .models import MFModel, RIOTModel, DeepMatrixtModel, CLIDIOTModel, CLIDIOTGradientModel
from .logger import console
from .constants import device, SEED
from .utils import seed


def get_mf_model(dataset, source, verbose=True):
    assert source in ['train', 'load']

    (X_train, Y_train), (X_test, Y_test), (m, n), (user_id_map, movie_id_map) = get_dataset(dataset)
    mf_model = MFModel(m=m, n=n, k=10 if dataset == 'small' else 15)
    if source == 'load':
        mf_model.load(dataset)
    else:
        mf_model.fit(X_train, Y_train, X_test, Y_test,
                     early_stopping=True, n_epochs=2000, l2=1e-2, lr=2, batch_size=None)
    if verbose:
        score = mf_model.score(X_test, Y_test)
        console.log(f'MF score: {score:.4f}')
    return mf_model


def get_deepmf_model(dataset, source, verbose=True):
    assert source in ['train', 'load']

    (X_train, Y_train), (X_test, Y_test), (m, n), (user_id_map, movie_id_map) = get_dataset(dataset)
    deepmf_model = DeepMatrixtModel(nb_users=m, nb_movies=n, k=20)

    if source == 'load':
        deepmf_model.load(dataset)
    else:
        deepmf_model.fit(
            (X_train, Y_train), (X_test, Y_test),
            opti='adam', steps=5, verbose=True,
            batch_size=1024 if dataset == 'small' else 50000
        )
    if verbose:
        score = deepmf_model.score((X_train, Y_train), (X_test, Y_test))
        console.log(f'DeepMF score: {score:.4f}')
    return deepmf_model


def main():
    seed()
    DATASET = 'large'

    console.log(f'random seed: {SEED}, device: {device}')

    with console.status('Loading data...'):
        (X_train, Y_train), (X_test, Y_test), (m, n) = get_dataset(DATASET)

    console.log(
        f'Data loaded ({m} users, {n} movies and {len(X_train)} ratings).')
    data_tab = Table()
    data_tab.add_column('Data')
    data_tab.add_column('Shape')
    data_tab.add_row('X_train', str(X_train.shape))
    data_tab.add_row('Y_train', str(Y_train.shape))
    data_tab.add_row('X_test', str(X_test.shape))
    data_tab.add_row('Y_test', str(Y_test.shape))
    console.log(data_tab)

    # features_model = get_mf_model(DATASET, 'load')
    features_model = get_deepmf_model(DATASET, 'load')

    U, V = features_model.get_features(X_train, Y_train)

    clidiot_model = CLIDIOTGradientModel(m, n)
    clidiot_model.fit(X_train, Y_train, X_test, Y_test, U=U, V=V,
                      eps=1e-4, max_iter=100, lr=1e-2, optimizer='adam')
    score = clidiot_model.score(X_test, Y_test)
    console.log(f'CLIDIOT score: {score:.4f}')

    # U_np, V_np = U.cpu().numpy(), V.cpu().numpy()
    # riot_model = RIOTModel(m, n, U_np, V_np)
    # riot_model.fit(X_train, Y_train)
    # score = riot_model.score(X_test, Y_test)
    # console.log(f'RIOT score: {score:.4f}')

    # clidiot_model = CLIDIOTModel(m, n)
    # clidiot_model.fit(X_train, Y_train, X_test, Y_test, U=U, V=V, eps=1e1)
    # score = clidiot_model.score(X_test, Y_test)
    # console.log(f'CLIDIOT score: {score:.4f}')
