import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LogLocator

from base import get_data1
from solutions import gradient_descent, compute_cost

data = pd.read_csv("ex1data1.txt", header=None).to_numpy()

plt.plot(data[:, 0], data[:, 1], 'rx', markersize=10)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

plt.show()

X, y, m = get_data1()
iterations = 1500
alpha = 0.01

theta, J_history = gradient_descent(X, y, np.zeros(2), alpha, iterations)

plt.plot(X[:, 1], y, 'rx', markersize=10)
plt.plot(X[:, 1], X@theta, '-')
plt.show()


# Surface plot of the cost function

theta0_values = np.arange(-10, 10.1, 0.2)
theta1_values = np.arange(-1, 4.01, 0.05)

t0, t1 = np.meshgrid(theta0_values, theta1_values)
J = []
for a, b in zip(np.nditer(t0), np.nditer(t1)):
    J.append(compute_cost(X, y, np.array([a, b])))
J = np.array(J).reshape(t0.shape)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot_surface(t0, t1, J, cmap=cm.coolwarm)
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")

plt.show()

# Contour plot of the cost function

plt.contour(t0, t1, J, cmap=cm.coolwarm, locator=LogLocator(2))
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show()
