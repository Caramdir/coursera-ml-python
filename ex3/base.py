from typing import *
import pathlib
import math

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def get_data1() -> Tuple[np.ndarray, np.ndarray, int]:
    data = scipy.io.loadmat(pathlib.Path(__file__).parent.absolute().joinpath("ex3data1.mat"))
    y = data["y"].flatten()
    y[np.nonzero(y == 10)] = 0  # Reindex to 0-based arrays
    return data["X"], y, data["X"].shape[0]


def display_data(X: np.ndarray, example_width: int = None):
    """function [h, display_array] = displayData(X, example_width)
    %DISPLAYDATA Display 2D data in a nice grid
    %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    %   stored in X in a nice grid. It returns the figure handle h and the
    %   displayed array if requested.
    """
    m, n = X.shape

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = round(math.sqrt(n))
    example_height = round(n / example_width)

    # Compute number of items to display
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))

            # Copy the patch (note that we need to transpose the data)
            display_array[
                          np.ix_(pad + j * (example_height + pad) + np.arange(example_height),
                                 pad + i * (example_width + pad) + np.arange(example_width))
                         ] = X[curr_ex, :].reshape(example_height, example_width).T / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break

    plt.imshow(display_array, cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")
    plt.show()


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function

    Computes the sigmoid of z.

    :param z: Input value(s).
    :return: Sigmoid function applied to each element of `z`.
    """
    return 1. / (1 + np.exp(-z))


def get_weights() -> Tuple[np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(pathlib.Path(__file__).parent.absolute().joinpath("ex3weights.mat"))

    # Reindex by moving the last column to the front.
    Theta2: np.ndarray = data["Theta2"]
    Theta2 = np.vstack((Theta2[-1], Theta2[:-1]))

    return data["Theta1"], Theta2
