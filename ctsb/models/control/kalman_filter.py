"""
Kalman Filter
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.models.control import ControlModel

class KalmanFilter(ControlModel):
    """
    Description: Kalman Filter adjusts measurements of a signal based on prior states and
    knowledge of intrinsic equations of the system.

    More precisely, we know that the signal at time t is a linear combination
    of its previous value plus a control signal u(t) and a process noise
    w(t - 1), i.e. x(t) = A x(t - 1) + B u(t) + w(t), and that the
    measurement at time t is a linear combination of the signal value and
    the measurement noise v(t), i.e. z(t) = H x(t) + v(t).

    Based on these, the model can advance by itself in time using a 'time'
    update and/or incorporate and correct a measurement using a 'measurement'
    update:

    a. Time Update (prediction)
    - Project state ahead: x(t) = A x(t - 1) + B u(t)
    - Project error covariance ahead: P(t) = A P(t - 1) A^T + Q

    b. Measurement Update
    - Compute Kalman Gain: K(t) = P(t) H^T (H P(t) H^T + R)^{-1}
    - Update estimate based on measurement: x(t) = x(t) + K(t) (z(t) - H x(t))
    - Update error covariance: P(t) = (I - K(t) H) P(t)

    where we assume w(t) ~ N(0, Q) and v(t) ~ N(0, R).

    The user must provide estimates for A, B, H, Q and R, as well as initial
    estimates for x(0) and P(0).
    """

    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x_2D = x[None, None]
        return x_2D

    def initialize(self, x, A, B, H, P, Q, R):
        """
        Description:
            Initialize the dynamics of the model.
        Args:
            x (float/numpy.ndarray): estimate of x(0)
            A (float/numpy.ndarray): past value contribution coefficient
            B (float/numpy.ndarray): control signal contribution coefficient
            H (float/numpy.ndarray): true signal contribution coefficient
            P (float/numpy.ndarray): initial estimate of error covariance P(0)
            Q (float/numpy.ndarray): covariance of model noise w(t)
            R (float/numpy.ndarray): covariance of environment noise v(t)
        Returns:
            None
        """

        self.initialized = True

        x, A, B, H, P, Q, R = self.to_ndarray(x), self.to_ndarray(A), self.to_ndarray(B), self.to_ndarray(H), self.to_ndarray(P), self.to_ndarray(Q), self.to_ndarray(R)

        self.x, self.A, self.B, self.H, self.P, self.Q, self.R  = x, A, B, H, P, Q, R
        self.K = np.ndarray(A.shape)

    def step(self, u, z, n = 1):
        """
        Description:
            Takes input measurement and control signal at current time-step,
            updates internal parameters, then returns the corresponding
            estimated true value.
        Args:
            u (float/numpy.ndarray): control signal at current time-step
            z (float/numpy.ndarray): measurement at current time-step
            n (non-negative int): number of 'time' updates before
                                 'measurement' update
        Returns:
            Estimated true value
        """

        u, z = self.to_ndarray(u), self.to_ndarray(z)

        for i in range(n):
            # time update
            self.x = self.A @ self.x + self.B @ u
            self.P = self.A @ self.P @ self.A.T + self.Q

        # measurement update
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + self.K @ (z - self.H @ self.x)
        self.P = self.P - self.K @ self.H @ self.P

        if(type(z) is float):
            return float(self.x)
        else:
            return self.x

    def predict(self, u, z, n = 1):
        """
        Description:
            Takes input measurement and control signal at current time-step,
            and returns the corresponding estimated true value
        Args:
            u (float/numpy.ndarray): control signal at current time-step
            z (float/numpy.ndarray): measurement at current time-step
        Returns:
            Estimated true value
        """

        u, z = self.to_ndarray(u), self.to_ndarray(z)

        for i in range(n):
            # time update
            x_temp = self.A @ self.x + self.B @ u
            P_temp = self.A @ self.P @ self.A.T + self.Q

        # measurement update
        K_temp = P_temp @ self.H.T @ np.linalg.inv(self.H @ P_temp @ self.H.T + self.R)
        x_temp = x_temp + K_temp @ (z - self.H @ x_temp)

        if(type(z) is not np.ndarray):
            return float(x_temp)
        else:
            return x_temp


    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(KalmanFilter_help)

    def __str__(self):
        return "<KalmanFilter Model>"


# string to print when calling help() method
KalmanFilter_help = """

-------------------- *** --------------------

Id: KalmanFilter

Description:

    Kalman Filter adjusts measurements of a signal based on prior states and
    knowledge of intrinsic equations of the system.

    More precisely, we know that the signal at time t is a linear combination
    of its previous value plus a control signal u(t) and a process noise
    w(t - 1), i.e. x(t) = A x(t - 1) + B u(t) + w(t), and that the
    measurement at time t is a linear combination of the signal value and
    the measurement noise v(t), i.e. z(t) = H x(t) + v(t).

    Based on these, the model can advance by itself in time using a 'time'
    update and/or incorporate and correct a measurement using a 'measurement'
    update:

    a. Time Update (prediction)
    - Project state ahead: x(t) = A x(t - 1) + u(t)
    - Project error covariance ahead: P(t) = A P(t - 1) A^T + Q

    b. Measurement Update
    - Compute Kalman Gain: K(t) = P(t) H^T (H P(t) H^T + R)^{-1}
    - Update estimate based on measurement: x(t) = x(t) + K(t) (z(t) - H x(t))
    - Update error covariance: P(t) = (I - K(t) H) P(t)

    where we assume w(t) ~ N(0, Q) and v(t) ~ N(0, R).

    The user must provides estimates for A, B, H, Q and R, as well as initial
    estimates for x(0) and P(0).

Methods:

    initialize(x, A, B, H, P, Q, R)
        Description:
            Initialize the dynamics of the model.
        Args:
            x (float/numpy.ndarray): estimate of x(0)
            A (float/numpy.ndarray): past value contribution coefficient
            B (float/numpy.ndarray): control signal contribution coefficient
            H (float/numpy.ndarray): true signal contribution coefficient
            P (float/numpy.ndarray): initial estimate of error covariance P(0)
            Q (float/numpy.ndarray): covariance of model noise w(t)
            R (float/numpy.ndarray): covariance of environment noise v(t)
        Returns:
            None

    step(x)
        Description:
            Takes input measurement and returns the corresponding estimated true value,
            then updates internal parameters
        Args:
            z (float/numpy.ndarray): easurement at current time-step
        Returns:
            Estimated true value

    predict(x)
        Description:
            Takes input measurement and returns the corresponding estimated true value
        Args:
            z (float/numpy.ndarray): measurement at current time-step
        Returns:
            Estimated true value

    update(rule=None)
        Description:
            Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
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