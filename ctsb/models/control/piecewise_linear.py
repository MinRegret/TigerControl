"""
Finds the optimal trajectory to swing-up a pendulum cart.
"""

import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class PiecewiseLinear(ControlModel):
    """
    Solves the trajectory on a uniform grid, approximating the trajectory
    (both state and action) as piecewise linear functions. Defects for
    collocation are computed by comparing the state at time k + 1 against
    the 4th order Runge-Kutta integration step from state at k.
    """

    def __init__(self):
        self.initialized = False

    def set_parameters(self):
        """
        Description:
            This function returns all user-sepcified parameters as a struct and the
            initial guess for the decision variables in the optimization.
        Args:
            None
        Returns:
            P (struct): A struct with a field for each set of parameters
            X (struct): Initial guess for decision variables
        """

        # Parameters for multiple shooting

        P.MS.nGrid = 100    # number of grid points to use
        P.MS.Start = [0;0;pi;0] # initial state [x,v,th,w]
        P.MS.Finish = [0;0;0;0]; # Final state [x,v,th,w]

        P.MS.solver = 'fmincon' # 'fmincon' solver

        # Cost function:
        P.MS.cost_function = 1 # 0 = none, 1 = force squared

        # FMINCON STUFF
        # ???

        # Bounds on the decision variables
        # Note - the state and actuator bounds are only enforced at the grid
        # points. It is possible for the actual values of the solution to exceed
        # these bounds.

        P.Bnd.force = 100*[-1,1];  #Bounds on the horizontal actuator

        P.Bnd.state = [-2, 2;    # Bounds on horizontal position of the cart
                      -10,10;    # Bouds on the horizontal velocity of the cart
                      -3*pi, 3*pi;  # Bounds on the pendulum angle
                    -2*2*pi, 2*2*pi]  # Bounds on the pendulum angular velocity

        P.Bnd.duration = [0.01,2]

        P.Dyn.m = 0.8   # Mass of the pendulum
        P.Dyn.M = 1   # Mass of the cart
        P.Dyn.g = 9.81   # Set negative to have 0 be the unstable equilibrium
        P.Dyn.L = 0.3   # Length of the pendulum

        P.Dyn.dynamics_func = @(x,f)Pendulum_Cart_Dynamics_Forced(x,f,P.Dyn);


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

        #Set up the problem - parameters, initial guess, decision variable bounds
        [P, X] = Set_Parameters();
        [Xlow, Xupp] = Set_Bounds(P);

    def step(self, n = 1):
        """
        Description:
            Try n new trajectories and return the current best one
        Args:
            n (non-negative int): number of new trajectories to try
        Returns:
            Current best trajectory
        """

        Results = wrapper_FMINCON(P,X,Xlow,Xupp);

        Xsoln = Results.Xsoln;
        State = [P.MS.Start, Xsoln.state, P.MS.Finish];
        Time = linspace(0,Xsoln.duration,P.MS.nGrid);
        Force = Xsoln.force;

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

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(PiecewiseLinear_help)

    def __str__(self):
        return "<PiecewiseLinear Model>"


# string to print when calling help() method
PiecewiseLinear_help = """

-------------------- *** --------------------

Id: PiecewiseLinear

Description: Solves the trajectory on a uniform grid, approximating the trajectory
             (both state and action) as piecewise linear functions. Defects for
             collocation are computed by comparing the state at time k + 1 against
             the 4th order Runge-Kutta integration step from state at k.

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