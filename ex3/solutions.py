import numpy as np
from scipy.optimize import minimize

from base import sigmoid


def lr_cost_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l: float) -> float:
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


def lr_cost_function_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l: float) -> np.ndarray:
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


def one_vs_all(X: np.ndarray, y: np.ndarray, num_labels: int, l: float) -> np.ndarray:
    """
    Trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i.

    Trains num_labels
    logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds
    to the classifier for label i.

    NOTE: The labels are indexed 0,...,num_labels-1 (rather than
          1,...,num_labels in the Matlab exercises).

    :param X: features
    :param y: targets
    :param num_labels: number of labels (y is in range(num_labels))
    :param l: regularization parameter lamba
    :return: trained parameters
    """

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter l.

    # =============================================================

    return all_theta


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier. The labels
    are in the range 0..K-1, where K = size(all_theta, 1).

    Will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class. You should set p to a vector
    of values from 0..K-1 (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    for 4 examples)

    :param all_theta: Trained parameters
    :param X: observations
    :return: predictions
    """
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    #
    # Hint: This code can be done all vectorized using the np.argmax method.

    # =========================================================================
    return p


def predict(Theta1: np.ndarray, Theta2:np.ndarray, X:np.ndarray) -> np.ndarray:
    """
    Predict the label of an input given a trained neural network.

    Outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)

    :param Theta1: Parameters for first layer
    :param Theta2: Parameters for second layer
    :param X: observations
    :return: predictions
    """

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 0 to num_labels-1.

    # =========================================================================

    return p
