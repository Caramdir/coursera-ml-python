import unittest

import numpy as np

from base import get_data1
from solutions import lr_cost_function, lr_cost_function_grad


class Exercise3TestCase(unittest.TestCase):
    def test_lr_cost_function(self):
        X, y, m = get_data1()
        theta_t = np.array([-2, -1, 1, 2])
        X_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order="F")/10))
        y_t = np.array([1, 0, 1, 0, 1])
        lambda_t = 3

        np.testing.assert_allclose(lr_cost_function(theta_t, X_t, y_t, lambda_t), 2.534819, atol=0.0000005)
        np.testing.assert_allclose(lr_cost_function_grad(theta_t, X_t, y_t, lambda_t)[:5],
                                   [0.146561, -0.548558, 0.724722, 1.398003], atol=0.0000005)


if __name__ == '__main__':
    unittest.main()
