import pathlib


import numpy as np
import pandas as pd

from typing import *


def get_data1() -> Tuple[np.ndarray, np.ndarray, int]:
    data = pd.read_csv(pathlib.Path(__file__).parent.absolute().joinpath("ex1data1.txt"), header=None).to_numpy()
    y = data[:, 1].copy()
    m = len(y)
    X = np.hstack((np.ones((m, 1)), data[:, 0].copy().reshape(-1, 1)))

    return X, y, m


def get_data2() -> Tuple[np.ndarray, np.ndarray, int]:
    data = pd.read_csv(pathlib.Path(__file__).parent.absolute().joinpath("ex1data2.txt"), header=None).to_numpy()

    y = data[:, 2].copy()
    m = len(y)
    # X = np.hstack((np.ones((m, 1)), data[:, 0:2].copy()))
    X = data[:, 0:2].copy()

    return X, y, m
