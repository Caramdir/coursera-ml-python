import numpy as np

from base import get_data1, display_data
from solutions import one_vs_all, predict_one_vs_all

X, y, m = get_data1()

display_data(X[np.random.default_rng().permutation(m)[0:100], :])

print('Training One-vs-All Logistic Regression...')

l = 0.1
all_theta = one_vs_all(X, y, 10, l)

p = predict_one_vs_all(all_theta, X)

print('Training Set Accuracy: ', np.mean((p == y).astype('float64')) * 100)
