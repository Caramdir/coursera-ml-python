import numpy as np
from scipy.optimize import minimize

from base import sigmoid


def nn_cost_function(nn_params: np.ndarray, input_layer_size: int,
                     hidden_layer_size: int, num_labels: int,
                     X: np.ndarray, y: np.ndarray, l: float) -> float:
    """
    Implements the neural network cost function for a two layer
    neural network which performs classification.

    Computes the cost of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    :param nn_params: unrolled parameters
    :param input_layer_size: input layer size (without bias)
    :param hidden_layer_size: hidden layer size (without bias)
    :param num_labels: number of output labels
    :param X: features
    :param y: targets (labeled 0,..., num_labels-1)
    :param l: regularization parameter
    :return: value of cost function
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size+1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size+1))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variable correctly
    J = 0

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return J


def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
    """
    Returns the gradient of the sigmoid function evaluated at z.

    Computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    """

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).

    # =============================================================
    return g


def nn_cost_function_grad(nn_params: np.ndarray, input_layer_size: int,
                          hidden_layer_size: int, num_labels: int,
                          X: np.ndarray, y: np.ndarray, l: float) -> np.ndarray:
    """
    Implements the neural network cost function gradient for a two layer
    neural network which performs classification.

    Computes the gradient of the cost of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.

    :param nn_params: unrolled parameters
    :param input_layer_size: input layer size (without bias)
    :param hidden_layer_size: hidden layer size (without bias)
    :param num_labels: number of output labels
    :param X: features
    :param y: targets (labeled 0,..., num_labels-1)
    :param l: regularization parameter
    :return: value of cost function
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size+1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size+1))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variable correctly
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return np.append(Theta1_grad.flat, Theta2_grad.flat)


def rand_initialize_weights(L_in: int, L_out: int) -> np.ndarray:
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms

    :param L_in: number of outgoing connections
    :param L_out: number of incoming connections
    :return: random weight matrix
    """

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first column of W corresponds to the parameters for the bias unit
    #

    # =========================================================================

    return W

