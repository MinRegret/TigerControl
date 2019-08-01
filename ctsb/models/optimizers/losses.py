# loss functions

import jax.numpy as np 

L_D = 100

def boost_mse(y_pred, y_true):
    ''' Description: mean-square-error loss
        Args:
            y_pred : value predicted by model
            y_true : ground truth value
            eps: some scalar
    '''

    return 1. / L_D * np.dot(2 * y_pred - 2 * y_true, y_true)

def mse(y_pred, y_true):
    ''' Description: mean-square-error loss
        Args:
            y_pred : value predicted by model
            y_true : ground truth value
            eps: some scalar
    '''
    return np.mean((y_pred - y_true)**2)
    
def cross_entropy(y_pred, y_true, eps=1e-9):
    ''' Description: cross entropy loss, y_pred is equivalent to logits and y_true to labels
        Args:
            y_pred : value predicted by model
            y_true : ground truth value
            eps: some scalar
    '''
    return - np.mean(y_true * np.log(y_pred + eps))