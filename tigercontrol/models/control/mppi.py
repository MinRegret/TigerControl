"""
MPPI
"""

import jax
import jax.numpy as np
import jax.random as random
from tigercontrol.utils import generate_key
import tigercontrol
from tigercontrol.models.control import ControlModel

class MPPI(ControlModel):
    """
    Description: Implements Model Predictive Path Integral Control to compute optimal control sequence.
    """

    compatibles = set(['PyBullet'])

    def __init__(self):
        self.initialized = False

    def initialize(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1):
        """
        Description: Initialize the dynamics of the model.
        Args:
            env (problem): The problem instance
            K (non-negative int): Number of trajectory samples
            T (non-negative int): Number of time steps
            U (array): Initial control sequence
            lambda_ (float): Scaling to ensure non-zero cost
            noise_mu (float): Mean of perturbation
            noise_sigma (float): Variance of perturbation
            u_init (float): Initial action
        """
        self.initialized = True

        self.K = K 
        self.T = T 
        
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        self.env = env

        self.x_init = self.env.getState()

        self.noise = (random.normal(generate_key(), shape=(self.K, self.T))) * noise_sigma + noise_mu

        def _ensure_non_zero(self, cost, beta, factor):
            return np.exp(-factor * (cost - beta))

        def _update():
            for k in range(self.K):
                self.compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            self.U += self.noise.T @ omega

            self.env.env.state = self.x_init
            s, r, _, _ = self.env.step([self.U[0]])
            self.env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U = jax.ops.index_update(self.U, -1, self.u_init)

            self.cost_total = np.zeros(self.cost_total.shape)

            self.x_init = self.env.getState()
            return

        self._ensure_non_zero = jax.jit(_ensure_non_zero)
        self._update = jax.jit(_update)

    def compute_total_cost(self, k):
        self.env.env.state = self.x_init
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            _, reward, _, _ = self.env.step([perturbed_action_t])
            self.cost_total = jax.ops.index_update(self.cost_total, k, self.cost_total[k] - reward)

    def plan(self, n = 100):
        """
        Description: Updates internal parameters and then returns the estimated optimal set of actions
        Args:
            n (non-negative int): Number of updates
        Returns:
            Estimated optimal set of actions
        """

        for i in range(n):
            self._update()

        return self.U

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(MPPI_help)

    def __str__(self):
        return "<MPPI Model>"


# string to print when calling help() method
MPPI_help = """

-------------------- *** --------------------

Id: MPPI

Description: Implements Model Predictive Path Integral Control to compute optimal control sequence.

Methods:

    initialize(env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1)
        Description:
            Initialize the dynamics of the model
        Args:
            env (problem): The problem instance
            K (non-negative int): Number of trajectory samples
            T (non-negative int): Number of time steps
            U (array): Initial control sequence
            lambda_ (float): Scaling to ensure non-zero cost
            noise_mu (float): Mean of perturbation
            noise_sigma (float): Variance of perturbation
            u_init (float): Initial action

    step()
        Description: Updates internal parameters and then returns the
            estimated optimal set of actions
        Args:
            n (non-negative int): Number of updates
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
            n (non-negative int): Number of updates

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""