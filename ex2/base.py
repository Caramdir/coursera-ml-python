import pathlib

import numpy as np
import pandas as pd

from typing import *


def get_data1() -> Tuple[np.ndarray, np.ndarray, int, int]:
    data = pd.read_csv(pathlib.Path(__file__).parent.absolute().joinpath("ex2data1.txt"), header=None).to_numpy()
    y = data[:, 2].copy()
    X = data[:, 0:2].copy()
    m, n = X.shape

    return X, y, m, n


def get_data1_with_intercept() -> Tuple[np.ndarray, np.ndarray, int, int]:
    X, y, m, n = get_data1()
    X = np.hstack((np.ones((m, 1)), X))
    return X, y, m, n


def get_data2() -> Tuple[np.ndarray, np.ndarray, int]:
    data = pd.read_csv(pathlib.Path(__file__).parent.absolute().joinpath("ex2data2.txt"), header=None).to_numpy()
    y = data[:, 2].copy()
    X = data[:, 0:2].copy()
    m = len(y)

    return X, y, m


def map_feature(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Feature mapping function to polynomial features

    Maps the two input features
    to the monomial features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    1, X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size

    :param X1: Feature 1
    :param X2: Feature 2
    :return: Up to sixth degree terms in features.
    """

    degree = 6
    out = np.ones((len(X1), 1))
    for i in range(degree):
        for j in range(i+2):
            out = np.append(out, ((X1**(i+1-j))*(X2**j)).reshape(-1, 1), axis=1)

    return out
