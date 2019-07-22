''' Metrics for evaluation '''

import jax.numpy as np 

# mean-square-error loss
def mse(y_pred, y_true):
    return np.sum((y_pred - y_true)**2)

# cross entropy loss, y_pred is equivalent to logits and y_true to labels
def cross_entropy(y_pred, y_true, eps=1e-9):
    return - np.dot(y_true, np.log(y_pred + eps))