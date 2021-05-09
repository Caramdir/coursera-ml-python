import numpy as np

from base import get_data1, get_weights
from solutions import predict

X, y, m = get_data1()
Theta1, Theta2 = get_weights()

p = predict(Theta1, Theta2, X)

print('Training Set Accuracy: ', np.mean((p == y).astype('float64')) * 100)
