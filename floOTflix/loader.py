import pandas as pd
import wget
import zipfile
import os
from typing import Tuple
import torch
import functools

from .constants import device

TDataSet = Tuple[torch.TensorType, torch.TensorType]
TDataShape = Tuple[int, int]

DATASET_PATH_NAMES = {
    'small': 'ml-latest-small',
    'large': 'ml-latest',
}


def _download_dataset(dataset):
    name = DATASET_PATH_NAMES[dataset]
    if os.path.exists(name):
        return name
    url = f'https://files.grouplens.org/datasets/movielens/{name}.zip'
    wget.download(url, f'{name}.zip')
    with zipfile.ZipFile(f'{name}.zip', 'r') as zip_ref:
        zip_ref.extractall()
    return name


def _load_dataset(folder_name: str):
    df = pd.read_csv(os.path.join(folder_name, 'ratings.csv'))

    df.userId -= 1
    df.movieId -= 1

    user_ids = df.userId.unique()
    movie_ids = df.movieId.unique()
    m = len(user_ids)
    n = len(movie_ids)

    # Create a mapping from user/movie ids to indices.
    # Using map because replace is too slow
    user_id_map = dict(zip(user_ids, range(m)))
    df.userId = df.userId.map(user_id_map)
    movie_id_map = dict(zip(movie_ids, range(n)))
    df.movieId = df.movieId.map(movie_id_map)

    # "remove users that have only one rating"
    df = df[df.rating > 1]

    # Select one random rating of each user and extract its index from df.
    selected_ratings_by_user = df.groupby('userId').apply(
        lambda x: x.sample(1)).index.get_level_values(1)

    # "testset: 1 random rating for each user in the dataset"
    test_ratings = df[df.index.isin(selected_ratings_by_user)]
    # "testtrain: all the other ratings"
    train_ratings = df[~df.index.isin(selected_ratings_by_user)]

    X_train = train_ratings[['userId', 'movieId']].to_numpy()
    y_train = train_ratings['rating'].to_numpy()

    # testset: 1 random rating for each user in the dataset
    X_test = test_ratings[['userId', 'movieId']].to_numpy()
    y_test = test_ratings['rating'].to_numpy()

    assert len(X_train) + len(X_test) == len(df), \
        "Train and test sets are not disjoint"

    return (m, n), (user_id_map, movie_id_map), (X_train, y_train), (X_test, y_test)


@functools.cache
def get_dataset(dataset) -> Tuple[TDataSet, TDataSet, TDataShape]:
    """Get the MovieLens dataset.

    Args:
        dataset (str): Name of the dataset (small, large)

    Returns:
        Tuple[TDataSet, TDataSet, TDataShape]: The train and test sets, and the shape of the data.
    """
    # folder_name = _download_dataset(dataset)
    shape, maps, *datasets = _load_dataset("ml-latest")
    datasets = ([
        torch.tensor(d[0].astype(int), dtype=int,
                     device=device, requires_grad=False),
        torch.tensor(d[1], device=device,
                     requires_grad=False, dtype=torch.float32)
    ] for d in datasets)
    return *datasets, shape, maps
