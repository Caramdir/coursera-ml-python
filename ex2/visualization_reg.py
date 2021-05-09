import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from base import get_data2
from solutions import plot_data

# Plot data

X, y, _ = get_data2()

plot_data(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend([r"$y=1$", r"$y=0$"])
plt.show()
