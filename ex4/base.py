from typing import *
import pathlib

import scipy.io
import numpy as np


def get_data() -> Tuple[np.ndarray, np.ndarray, int]:
    data = scipy.io.loadmat(pathlib.Path(__file__).parent.absolute().joinpath("ex4data1.mat"))
    y = data["y"].flatten()
    y[np.nonzero(y == 10)] = 0  # Reindex to 0-based arrays
    return data["X"], y, data["X"].shape[0]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function

    Computes the sigmoid of z.

    :param z: Input value(s).
    :return: Sigmoid function applied to each element of `z`.
    """
    return 1. / (1 + np.exp(-z))


def get_weights() -> Tuple[np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(pathlib.Path(__file__).parent.absolute().joinpath("ex4weights.mat"))

    # Reindex by moving the last column to the front.
    Theta2: np.ndarray = data["Theta2"]
    Theta2 = np.vstack((Theta2[-1], Theta2[:-1]))

    return data["Theta1"], Theta2


def debug_initialize_weights(fan_out: int, fan_in: int) -> np.ndarray:
    """
    Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging.

    Note that it should return a matrix of size (1 + fan_in, fan_out) as
    the first row handles the "bias" terms

    :param fan_out: number of outgoing connections
    :param fan_in: number of incoming connections
    :return: weight matrix
    """

    w = np.zeros((fan_out, 1 + fan_in))

    # Initialize w using "sin", this ensures that w is always of the same
    # values and will be useful for debugging
    return np.sin(np.arange(w.size)+1).reshape(w.shape) / 10


def predict(Theta1: np.ndarray, Theta2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predict the label of an input given a trained neural network.

    Outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)

    :param Theta1: Weights for first layer
    :param Theta2: Weights for second layer
    :param X: Observations
    :return: Prediction
    """

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    h1 = sigmoid(np.hstack((np.ones((m, 1)), X)) @ Theta1.T)
    h2 = sigmoid(np.hstack((np.ones((m, 1)), h1)) @ Theta2.T)

    return np.argmax(sigmoid(h2), axis=1)
