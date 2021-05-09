import numpy as np
import matplotlib.pyplot as plt

from base import get_data2
from solutions import feature_normalize, gradient_descent_multi, predict, normal_eqn, predict_normal_eqn


X, y, m = get_data2()

X, mu, sigma = feature_normalize(X)

X = np.hstack((np.ones((m, 1)), X))

alpha = 0.01
num_iters = 400

theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel(r'Cost $J$')
plt.show()

print('Theta computed from gradient descent:')
print(theta)
print('')

price = predict(np.array([1650, 3]), mu, sigma, theta)

print('Predicted price of a 1650 sq-ft, 3 br house ',
      "(using gradient descent):\n", price, "\n")


X, y, m = get_data2()
X = np.hstack((np.ones((m, 1)), X))

theta = normal_eqn(X, y)

print('Theta computed from the normal equations: \n')
print(theta)
print()

price = predict_normal_eqn(np.array([1650, 3]), theta)

print('Predicted price of a 1650 sq-ft, 3 br house ',
      "(using normal equations):\n", price, "\n")

