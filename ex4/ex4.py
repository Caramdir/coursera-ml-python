import unittest

import numpy as np
from scipy.optimize import check_grad

from base import get_weights, get_data, debug_initialize_weights
from solutions import nn_cost_function, sigmoid_gradient, nn_cost_function_grad


class NNTestCase(unittest.TestCase):
    def test_cost_function(self):
        X, y, m = get_data()
        Theta1, Theta2 = get_weights()
        nn_params = np.append(Theta1.flat, Theta2.flat)

        cost = nn_cost_function(nn_params, Theta1.shape[1]-1, Theta1.shape[0], Theta2.shape[0], X, y, 0)
        np.testing.assert_allclose(cost, 0.287629, atol=0.0000005)

        cost = nn_cost_function(nn_params, Theta1.shape[1]-1, Theta1.shape[0], Theta2.shape[0], X, y, 1)
        np.testing.assert_allclose(cost, 0.383770, atol=0.0000005)

    def test_sigmoid_gradient(self):
        np.testing.assert_allclose(sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1])),
                                   [0.196612, 0.235004, 0.250000, 0.235004, 0.196612], atol=0.0000005)

    def test_cost_function_gradient(self):
        l = 0
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        # We generate some 'random' test data
        Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
        Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
        # Reusing debug_initialize_weights to generate X
        X = debug_initialize_weights(m, input_layer_size - 1)
        y = np.arange(m) % num_labels

        nn_params = np.append(Theta1.flat, Theta2.flat)

        args = (Theta1.shape[1]-1, Theta1.shape[0], Theta2.shape[0], X, y, 0)
        np.testing.assert_allclose(
            0,
            check_grad(nn_cost_function, nn_cost_function_grad, nn_params, *args),
            atol=0.000001
        )

        args = (Theta1.shape[1]-1, Theta1.shape[0], Theta2.shape[0], X, y, 3)
        np.testing.assert_allclose(
            0,
            check_grad(nn_cost_function, nn_cost_function_grad, nn_params, *args),
            atol=0.000001
        )


if __name__ == '__main__':
    unittest.main()
