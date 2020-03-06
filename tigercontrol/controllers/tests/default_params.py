"""
default controller init parameters
"""
import tigercontrol
import jax.numpy as np

""" mapping from controller ids to default initialization parameters """
control_params = {
    'GPC': {
        'A': np.identity(3)
        'B': np.identity(3)
    }
    'BPC': {
        'A': np.identity(3)
        'B': np.identity(3)
    }
    'LQR': {
        'A': np.identity(3)
        'B': np.identity(3)
    }
    'ILQR': {
        'env': tigercontrol.controller("Cartpole")() # cartpole instance
    }
    'SimpleBoost': {
        'A': np.identity(3)
        'B': np.identity(3)
    }
}
