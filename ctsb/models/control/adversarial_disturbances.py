"""
Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel
from ctsb.models.control import ILQR
from ctsb import error

class AdversarialDisturbances(ControlModel):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False


    def initialize(self, dynamics, L, dim_x, dim_u, H, ilqr_params):
        """
        Description: Initialize the dynamics of the model
        Args:
            dynamics (function *OR* tuple of matrices A,B): dynamics of problem
            L (function): loss function
            dim_x (int): state_space dimension
            dim_u (int): action_space dimension
            H (int): history length
            ilqr_params (dict): params to pass ILQR during planning
        """
        self.initialized = True
        self.linear_dynamics = (not callable(dynamics))
        if callable(dynamics):
            dyn_jacobian = jax.jit(jax.jacrev(dyn, argnums=(0,1)))
            self.get_dynamics = lambda x, u: dyn_jacobian(x, u)
        else:
            if len(dynamics) != 2:
                raise error.InvalidInput("dynamics input must either be a function with two arguments or a pair of matrices!")
            self.get_dynamics = lambda x, u: dynamics[0], dynamics[1]
        self.L = L
        self.H = H
        self.dim_x, self.dim_u = dim_x, dim_u
        self.noise = [np.zeros(self.dim_x) for i in range(H)]
        self.M = [np.zeros((self.dim_x, self.dim_x)) for i in range(H)]
        self.ilqr = ILQR()
        self.ilqr.initialize(dynamics, L, dim_x, dim_u)


    def plan(self, x_0, T, max_iterations=10, lamb=0.1, threshold=None):
        return


    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(AD_help)

    def __str__(self):
        return "<Adversarial Disturbances Model>"


# string to print when calling help() method
AD_help = """

-------------------- *** --------------------

Id: AdversarialDisturbances

Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.

Methods:

    initialize(F, f, C, c, T, x)
        Description:
            Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state

    step()
        Description: Updates internal parameters and then returns the
        	estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    predict()
        Description:
            Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    update()
        Description:
        	Updates internal parameters
        Args:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""