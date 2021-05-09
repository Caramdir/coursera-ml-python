import numpy as np
from typing import *


def warm_up_exercise() -> np.ndarray:
    """
    An example function that returns a 5x5 identity matrix.

    :return: A 5x5 identity matrix numpy array of type float.
    """
    return np.identity(5)


def compute_cost(X: np.array, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute cost for linear regression

    Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    :param X: observations
    :param y: targets
    :param theta: parameters
    :return: Sum of squares cost for given data and parameters.
    """
    m: int = len(y)     # number of training examples

    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should return the cost.

    # ============================================================

    return J


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iterations: int) -> Tuple[np.ndarray, List[float]]:
    """
    Performs gradient descent to learn `theta`

    Returns `theta` after taking `num_iterations` gradient steps with learning rate `alpha`.
    :param X: observations
    :param y: targets
    :param theta: learned parameter
    :param alpha: learning rate
    :param num_iterations: number of iterations to run
    :return: Updated value of `theta` and a list with the value of the cost function at each iteration.
    """
    m: int = len(y)     # number of training examples

    J_history: List[float] = []

    for i in range(num_iterations):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.

        # ============================================================
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def feature_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the features in X.

    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    :param X: features
    :return: A tuple consisting of the normalized `X`, and arrays
             containing old mean and the old standard deviation of
             each feature.
    """
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'np.mean' and 'np.std'
    #       functions useful.
    #

    # ============================================================

    return X_norm, mu, sigma


def gradient_descent_multi(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iterations: int) -> Tuple[np.ndarray, List[float]]:
    """
    Performs gradient descent to learn `theta`

    Returns `theta` after taking `num_iterations` gradient steps with learning rate `alpha`.
    :param X: observations
    :param y: targets
    :param theta: learned parameter
    :param alpha: learning rate
    :param num_iterations: number of iterations to run
    :return: Updated value of `theta` and a list with the value of the cost function at each iteration.
    """
    m: int = len(y)     # number of training examples

    J_history: List[float] = []

    for i in range(num_iterations):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.

        # ============================================================
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def predict(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, theta: np.ndarray):
    price = 0

    normalized_data = (x - mu) / sigma

    price = np.hstack((np.ones(1), normalized_data)) @ theta

    return price


def normal_eqn(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the closed-form solution to linear regression

    Computes the closed-form solution to linear
    regression using the normal equations.

    :param X: features
    :param y: targets
    :return: parameters for linear regression
    """

    theta = np.zeros(X.shape[1])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    # ============================================================
    return theta


def predict_normal_eqn(x: np.ndarray, theta: np.ndarray):
    price = 0

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return price
