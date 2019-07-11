"""
Shooting Method
"""

import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class ShootingMethod(ControlModel):
    """
    Implements the shooting method.
    """

    def __init__(self):
        self.initialized = False

    def trial(self, sim, x, target, T):
        """
        Description: Try out a random trajectory starting from the given state.
        Args:
            sim (simulator) : environment simulator
            x (float / numpy.ndarray): initial state
            target (float / numpy.ndarray): target state
            T (int): number of actions to plan
        Returns:
            Tuple containing a random trajectory and its distance from the target.
        """

        raise NotImplementedError

    def initialize(self, sim, x, target, T = 1):
        """
        Description: Initialize the dynamics of the model.
        Args:
            sim (simulator) : environment simulator
            x (float / numpy.ndarray): initial state
            target (float / numpy.ndarray): target state
            T (int): number of actions to plan
        """

        self.initialized = True

        self.sim, self.x, self.target, self.T = sim, x, target, T

        (self.u, self.dist) = self.trial(sim, x, target, T)

    def step(self, n = 1):
        """
        Description:
            Try n new trajectories and return the current best one
        Args:
            n (non-negative int): number of new trajectories to try
        Returns:
            Current best trajectory
        """

        for i in range(n):

            u, dist =  self.trial(self.sim, self.x, self.target, self.T)
            if (dist < self.dist):
                self.u, self.dist = u, dist

        return self.u

    def predict(self):
        """
        Description:
            Returns current best trajectory.
        Args:
            None
        Returns:
            Current best trajectory
        """

        return self.u


    def update(self, n = 1):
        """
        Description:
            Try n new trajectories and return the current best one
        Args:
            n (non-negative int): number of new trajectories to try
        Returns:
           None
        """

        for i in range(n):
            
            u, dist =  self.trial(self.sim, self.x, self.target, self.T)
            if (dist < self.dist):
                self.u, self.dist = u, dist

        return

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(ShootingMethod_help)

    def __str__(self):
        return "<ShootingMethod Model>"


# string to print when calling help() method
ShootingMethod_help = """

-------------------- *** --------------------

Id: ShootingMethod

Description: Implements the shooting method.

Methods:

    initialize(sim, x, target, T)
        Description: Initialize the dynamics of the model.
        Args:
            sim (simulator) : environment simulator
            x (float / numpy.ndarray): initial state
            target (float / numpy.ndarray): target state
            T (int): number of actions to plan

    step(n)
        Description:
            Try n new trajectories and return the current best one
        Args:
            n (non-negative int): number of new trajectories to try
        Returns:
            Current best trajectory

    predict()
        Description:
            Returns current best trajectory.
        Args:
            None
        Returns:
            Current best trajectory

    update(n)
        Description:
            Try n new trajectories and return the current best one
        Args:
            n (non-negative int): number of new trajectories to try
        Returns:
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