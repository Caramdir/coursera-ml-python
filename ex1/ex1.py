import unittest

import numpy as np

from base import get_data1
from solutions import warm_up_exercise, compute_cost, gradient_descent


class Exercise1TestCase(unittest.TestCase):
    def test_identity(self):
        self.assertTrue(np.array_equal(warm_up_exercise(), np.eye(5)))

    def test_compute_cost(self):
        X, y, m = get_data1()

        self.assertEqual(32.07, round(compute_cost(X, y, np.zeros(2)), 2))
        self.assertEqual(54.24, round(compute_cost(X, y, np.array([-1, 2])), 2))

    def test_gradient_descent(self):
        X, y, m = get_data1()

        iterations = 1500
        alpha = 0.01

        theta, _ = gradient_descent(X, y, np.zeros(2), alpha, iterations)
        np.testing.assert_allclose(theta, [-3.6303, 1.1664], atol=0.0001)


if __name__ == '__main__':
    unittest.main()
