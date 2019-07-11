"""
ODE Shooting Method
"""

import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class ODEShootingMethod(ControlModel):
    """
    Implements the shooting method to solve second order boundary value
    problems with conditions y(0) = a and y(L) = b. Assumes that the
    second order BVP has been converted to a first order system of two
    equations.
    """

    def __init__(self):
        self.initialized = False

    def euler(self, f, a, z, t, dt = 0.1):
        """
        Description:
            Solve corresponding initial value problem.
        Args:
            f (function): describes dy/dt = f(y,t)
            a (float): value of y(0)
            z (float): value of y'(0)
            t (float): time value to determine y at
            dt (float): stepsize
        Returns:
            Estimated solution function values at times specified in t
        """

        n = t / dt # compute number of iterations
        cur_t = 0

        for i in range(int(n)):

            z += dt * f(cur_t, a)
            a += dt * z
            cur_t += dt

        return z

    def initialize(self, f, a, b, z1, z2, t):
        """
        Description:
            Initialize the dynamics of the model.
        Args:
            f (function): describes dy/dt = f(y,t)
            a (float): value of y(0)
            b (float): value of y(L)
            z1 (float): first initial estimate of y'(0)
            z2 (float): second initial estimate of y'(0)
            t (float): time value to determine y at
        """

        self.initialized = True

        self.f, self.a, self.b, self.z1, self.z2, self.t = f, a, b, z1, z2, t

        self.w1 = self.euler(f, a, z1, t)
        self.w2 = self.euler(f, a, z2, t)

    def step(self, n = 1):
        """
        Description:
            Updates internal parameters for n iterations and then returns
            current solution estimation.
        Args:
            n (non-negative int): number of updates
        Returns:
            Estimated solution function values at times specified in t
        """

        for i in range(n):

            if(self.w1 == self.w2):
                break

            self.z1, self.z2 = self.z2, self.z2 + (self.z2 - self.z1) / (self.w2 - self.w1) * (self.b - self.w2)
            self.w1 = self.w2
            self.w2 = self.euler(self.f, self.a, self.z2, self.t)

        return self.w2

    def predict(self):
        """
        Description:
            Returns current solution estimation.
        Args:
            None
        Returns:
            Estimated solution function values at times specified in t
        """

        return self.w2


    def update(self):
        """
        Description:
            N / A
        Args:
            N / A
        Returns:
            N / A
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
        print(ShootingMethod_help)

    def __str__(self):
        return "<ODEShootingMethod Model>"


# string to print when calling help() method
ShootingMethod_help = """

-------------------- *** --------------------

Id: ODEShootingMethod

Description: Implements the shooting method to solve second order boundary value
            problems with conditions y(0) = a and y(L) = b. Assumes that the
            second order BVP has been converted to a first order system of two
            equations.

Methods:

    initialize(f, a, b, z1, z2, t)
        Description:
            Initialize the dynamics of the model.
        Args:
            f (function): describes dy/dt = f(y,t)
            a (float): value of y(0)
            b (float): value of y(L)
            z1 (float): first initial estimate of y'(0)
            z2 (float): second initial estimate of y'(0)
            t (float): time value to determine y at

    step(n)
        Description:
            Updates internal parameters for n iterations and then returns
            current solution estimation.
        Args:
            n (non-negative int): number of updates
        Returns:
            Estimated solution function values at times specified in t

    predict()
        Description:
            Returns current solution estimation.
        Args:
            None
        Returns:
            Estimated solution function values at times specified in t

    update()
        Description:
            N / A
        Args:
            N / A
        Returns:
            N / A

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""