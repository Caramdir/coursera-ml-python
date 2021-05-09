import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from base import get_data1, get_data1_with_intercept
from solutions import plot_data, cost_function, cost_function_grad, sigmoid, predict

# Plot data

X, y, _, _ = get_data1()

plot_data(X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(["Admitted", "Not admitted"])
plt.show()


# Plot decision boundary

X, y, m, n = get_data1_with_intercept()
initial_theta = np.zeros(n+1)

options = {'maxiter': 400}
minimum = minimize(cost_function, initial_theta, args=(X, y), jac=cost_function_grad, options=options)
theta = minimum.x

plot_data(X[:, 1:], y)
# Only need 2 points to define a line, so choose two endpoints
plot_x = np.array([min(X[:, 2])-2,  max(X[:, 2])+2])

# Calculate the decision boundary line
plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

# Plot
plt.plot(plot_x, plot_y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(["Admitted", "Not admitted", "Decision Boundary"])
plt.show()


# Predictions


prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with scores 45 and 85, we predict an admission ',
      'probability of ', prob)
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: ', np.mean((p == y).astype('float64')) * 100)
print('Expected accuracy (approx): 89.0')
