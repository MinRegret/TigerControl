"""
Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class iLQR(ControlModel):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
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
        if(np.ndim(x) == 0):
            x = x[None, None]
        return x

    def extend(self, x, T):
        """
        Description: If x is not in the correct form, convert it; otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
            T (postive int): number of timesteps
        Returns:
            A numpy.ndarray representation of x
        """
        x = self.to_ndarray(x)
        return [x for i in range(T)]

    def initialize(self, F, f, C, c, T, x):
        """
        Description: Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state
        """
        self.initialized = True
        self.dim_x, self.dim_u = dim_x, F.shape[1] - dim_x
        self.T = T
        self.x, self.u = self.to_ndarray(x), self.extend(np.zeros((self.dim_u, 1)), T)
        self.F, self.f, self.C, self.c = self.extend(F, T), self.extend(f, T), self.extend(C, T), self.extend(c, T)
        self.K = self.extend(np.zeros((self.dim_u, self.dim_x)), T)
        self.k = self.u.copy()


    def lqr(self, F, C, c, x, xs, T, lamb):
        """
        Description: Updates internal parameters and then returns the estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """
        dim_x, dim_u = self.dim_x, self.dim_u
        u = self.extend(np.zeros((dim_u, )), T)
        K = self.extend(np.zeros((dim_u, dim_x)), T)
        k = u.copy()
        ## Initialize V and Q Functions ##
        V = np.zeros((dim_x, dim_x))
        v = np.zeros((dim_x, ))
        Q = np.zeros((dim_x + dim_u, dim_x + dim_u))
        q = np.zeros((dim_x + dim_u, ))

        ## Backward Recursion ##
        for t in reversed(range(T)):

            Q = C[t] + F[t].T @ V @ F[t]
            q = c[t] + F[t].T @ v  # get rid of + F[t].T @ V @ f[t] in iLQR

            Q_uu, Q_ux, Q_xx = Q[dim_x:, dim_x:], Q[dim_x:, :dim_x], Q[:dim_x, :dim_x]
            q_u, q_x = q[dim_x:], q[:dim_x]
            Q_uu_evals, Q_uu_evecs = np.linalg.eigh(Q_uu)
            Q_uu_evals = lamb + np.maximum(Q_uu_evals, 0.0)
            Q_uu_inv = Q_uu_evecs @ np.diag(1. / Q_uu_evals) @ Q_uu_evecs.T

            K[t] = -Q_uu_inv @ Q_ux
            k[t] = -Q_uu_inv @ q_u

            V = Q_xx + Q_ux.T @ K[t] + K[t].T @ Q_ux + K[t].T @ Q_uu @ K[t]
            v = q_x + Q_ux.T @ k[t] + K[t].T @ q_u + K[t].T @ Q_uu @ k[t]

        ## Forward Recursion ##
        x_new = [xs[0]]
        u_new = [0 for i in range(T)]
        for t in range(T):
            u_new[t] = u[t] + k[t] + K[t] @ (x_new[t] - xs[t])            
            if t < T-1:
                x_new.append(self.dyn(x_new[t], u_new[t]))

        return u_new


    def ilqr(self, dim_u, dyn, L, x_0, T, threshold, lamb=0.1, max_iterations=10):
        # initialize
        self.dyn = dyn
        jacobian = jax.jacrev(dyn, argnums=(0,1))
        hessian = jax.hessian(L, argnums=(0,1))
        grad = jax.grad(L, argnums=(0,1))

        self.dim_x, self.dim_u = x_0.shape[0], dim_u
        u = self.extend(np.zeros((self.dim_u, )), T)

        count = 0
        while True:
            count += 1
            print("\ncount = " + str(count))
            if count > max_iterations:
                break
            
            x = [x_0]
            for i in range(T-1):
                x.append(dyn(x[-1],u[i]))

            #F = [jacobian(x[t],u[t][0]) for t in range(T)]
            #F = [np.hstack([F_x, F_u]) for (F_x, F_u) in F]
            F = [np.hstack(jacobian(x[t],u[t])) for t in range(T)]

            C = [hessian(x[t],u[t]) for t in range(T)]
            C = [np.vstack([np.hstack([C_x[0],C_x[1]]), np.hstack([C_u[0],C_u[1]])]) for (C_x, C_u) in C]

            c = [grad(x[t],u[t]) for t in range(T)]
            #c = [np.vstack([np.asarray([[y] for y in c_t[0]]), np.asarray(c_t[1])]) for c_t in c]
            c = [np.hstack([c_t[0], c_t[1]]) for c_t in c]
            print("c[0]: " + str(c[0]))

            #x_0_col = np.asarray([[y] for y in x_0])
            u_new = self.lqr(F, C, c, x_0, x, T, lamb)

            x_new = [x_0]
            for i in range(T-1):
                x_new.append(dyn(x_new[-1], u_new[i]))

            old_cost = np.sum([L(x[t],u[t]) for t in range(T)])
            new_cost = np.sum([L(x_new[t],u_new[t]) for t in range(T)])

            if new_cost < old_cost:
                u = u_new
                lamb = 2.0 * lamb
            else:
                lamb = 0.5 *lamb
            
            if abs(new_cost - old_cost) < threshold:
                break;

        return u


    def predict(self):
        """
        Description: Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """
        return self.u


    def update(self):
        """
        Description: Updates internal parameters
        Args:
            None
        """

        ## Initialize V and Q Functions ##
        V = np.zeros((self.self.dim_x, self.self.dim_x))
        v = np.zeros((self.self.dim_x, 1))
        Q = np.zeros((self.dim_x + self.dim_u, self.dim_x + self.dim_u))
        q = np.zeros((self.dim_x + self.dim_u, 1))

        ## Backward Recursion ##
        for t in range(self.T - 1, -1, -1):

            Q = self.C[t] + self.F[t].T @ V @ self.F[t]
            q = self.c[t] + self.F[t].T @ V @ self.f[t] + self.F[t].T @ v

            self.K[t] = -np.linalg.inv(Q[self.dim_x :, self.dim_x :]) @ Q[self.dim_x :, : self.dim_x]
            self.k[t] = -np.linalg.inv(Q[self.dim_x :, self.dim_x :]) @ q[self.dim_x :]

            V = Q[: self.dim_x, : self.dim_x] + Q[: self.dim_x, self.dim_x :] @ self.K[t] + self.K[t].T @ Q[self.dim_x :, : self.dim_x] + self.K[t].T @ Q[self.dim_x :, self.dim_x :] @ self.K[t]
            v = q[: self.dim_x] + Q[: self.dim_x, self.dim_x :] @ self.k[t] + self.K[t].T @ q[self.dim_x :] + self.K[t].T @ Q[self.dim_x :, self.dim_x :] @ self.k[t]

        ## Forward Recursion ##
        for t in range(self.T):
            self.u[t] = self.K[t] @ self.x + self.k[t]
            self.x = self.F[t] @ np.vstack((self.x, self.u[t])) + self.f[t]

        return

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LQR_help)

    def __str__(self):
        return "<LQR Model>"


# string to print when calling help() method
LQR_help = """

-------------------- *** --------------------

Id: LQR

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