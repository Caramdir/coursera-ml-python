import numpy as np
from scipy.optimize import minimize

from solutions import nn_cost_function, nn_cost_function_grad, rand_initialize_weights
from base import get_data, predict

input_layer_size  = 400     # 20x20 Input Images of Digits
hidden_layer_size = 25      # 25 hidden units
num_labels = 10             # 10 labels, from 0 to 9

X, y, m = get_data()

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_Theta1.flat, initial_Theta2.flat)

l = 1   # Regularization parameter lambda

options = {'maxiter': 50}
args = (input_layer_size, hidden_layer_size, num_labels, X, y, l)

print('Training Neural Network...')
result = minimize(nn_cost_function, initial_nn_params, args=args, method="CG", jac=nn_cost_function_grad, options=options)
print('Done')

Theta1 = result.x[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size+1))
Theta2 = result.x[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size+1))

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: ', np.mean((pred == y).astype('float64')) * 100)
