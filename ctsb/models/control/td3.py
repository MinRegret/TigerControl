"""
TD3 algorithm, Fujimoto et al. 2018 (https://arxiv.org/abs/1802.09477)
"""

import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class TD3(ControlModel):
    """
    Twin Delayed DDPG reinforcement learning algorithm.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self):
        """
        Description:
            Initialize the dynamics of the model.
        """

        self.initialized = True


    def step(self):
        """
        Description:
            Takes input measurement and control signal at current time-step,
            updates internal parameters, then returns the corresponding
            estimated true value.
        """
        return


    def predict(self):
        """
        Description:
            Takes input measurement and control signal at current time-step,
            and returns the corresponding estimated true value
        """
        return


    def update(self):
        """
        Description:
            Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None
        """
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
        print(TD3_help)

    def __str__(self):
        return "<TD3 Model>"


# string to print when calling help() method
TD3_help = """

-------------------- *** --------------------

Id: TD3

Description:

    Time Delayed DDPG

-------------------- *** --------------------

"""