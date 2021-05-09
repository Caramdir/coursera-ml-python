import numpy as np
import matplotlib.pyplot as plt


def plot_data(X: np.ndarray, y: np.ndarray):
    """
    Plots the data points X and y into a new figure.

    Plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    :param X: features
    :param y: classes
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #

    # =========================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function

    Computes the sigmoid of z.

    :param z: Input value(s).
    :return: Sigmoid function applied to each element of `z`.
    """

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).

    # =============================================================
    return g


def cost_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cost for logistic regression.

    Computes the cost of using theta as the parameter for logistic
    regression.

    :param theta: Parameters
    :param X: observations
    :param y: classes
    :return: cost
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variable correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.

    # =============================================================
    return J


def cost_function_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient for logistic regression

    Computes the gradient at theta of the cost function for
    logistic regression.
    :param theta: Parameters
    :param X: observations
    :param y: classes
    :return: gradient at `theta`
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variable correctly.
    grad = np.zeros(theta.size)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta.
    #
    # Note: grad should have the same dimensions as theta.
    #

    # =============================================================
    return grad


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters `theta`.

    Computes the predictions for `X` using a
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    :param theta: Parameters
    :param X: observations
    :return: predictions
    """

    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's

    # =========================================================================

    return p


def cost_function_reg(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l: float) -> float:
    """
    Compute cost for logistic regression with regularization.

    Computes the cost of using `theta` as the parameter for regularized
    logistic regression.

    :param theta: parameters
    :param X: features
    :param y: targets
    :param l: regularization parameter
    :return: cost for logarithmic regression
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variable correctly.
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.

    # =============================================================
    return J


def cost_function_reg_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l: float) -> np.ndarray:
    """
    Compute gradient for logistic regression with regularization.

    Computes the gradient at parameter theta for cost cost function
    for regularized logistic regression.

    :param theta: parameters
    :param X: features
    :param y: targets
    :param l: regularization parameter
    :return: gradient for logarithmic regression cost function
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variable correctly.
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions:  Compute the partial derivatives and set grad to the partial
    #                derivatives of the cost w.r.t. each parameter in theta.

    # =============================================================
    return grad
