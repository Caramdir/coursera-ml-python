import unittest

import numpy as np
from scipy.optimize import minimize

from base import get_data1_with_intercept, get_data2, map_feature

from solutions import cost_function, cost_function_grad, cost_function_reg, cost_function_reg_grad


class Exercise2TestCase(unittest.TestCase):
    def test_cost_function_zero_theta(self):
        X, y, m, n = get_data1_with_intercept()
        theta = np.zeros(n+1)

        np.testing.assert_allclose(cost_function(theta, X, y), 0.693, atol=0.0005)
        np.testing.assert_allclose(cost_function_grad(theta, X, y), [-0.1000, -12.0092, -11.2628], atol=0.00005)

    def test_cost_function_nonzero_theta(self):
        X, y, m, n = get_data1_with_intercept()
        theta = np.array([-24, 0.2, 0.2])

        np.testing.assert_allclose(cost_function(theta, X, y), 0.218, atol=0.0005)
        np.testing.assert_allclose(cost_function_grad(theta, X, y), [0.043, 2.566, 2.647], atol=0.0005)

    def test_optimization(self):
        X, y, m, n = get_data1_with_intercept()
        initial_theta = np.zeros(n+1)

        options = {'maxiter': 400}
        opt = minimize(cost_function, initial_theta, args=(X, y), jac=cost_function_grad, options=options)

        np.testing.assert_allclose(opt.fun, 0.203, atol=0.0005)
        np.testing.assert_allclose(opt.x, [-25.161, 0.206, 0.201], atol=0.0005)


class Exercise2RegularizationTestCase(unittest.TestCase):
    def test_cost_function_zero_theta(self):
        X, y, m = get_data2()
        X = map_feature(X[:, 0], X[:, 1])

        theta = np.zeros(X.shape[1])
        l = 1

        np.testing.assert_allclose(cost_function_reg(theta, X, y, l), 0.693, atol=0.0005)
        np.testing.assert_allclose(cost_function_reg_grad(theta, X, y, l)[:5],
                                   [0.0085, 0.0188, 0.0001, 0.0503, 0.0115], atol=0.00005)

    def test_cost_function_ones_theta(self):
        X, y, m = get_data2()
        X = map_feature(X[:, 0], X[:, 1])

        theta = np.ones(X.shape[1])
        l = 10

        np.testing.assert_allclose(cost_function_reg(theta, X, y, l), 3.16, atol=0.005)
        np.testing.assert_allclose(cost_function_reg_grad(theta, X, y, l)[:5],
                                   [0.3460, 0.1614, 0.1948, 0.2269, 0.0922], atol=0.00005)


if __name__ == '__main__':
    unittest.main()
