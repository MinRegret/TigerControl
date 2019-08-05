"""
Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

# for testing
import time

class iLQR(ControlModel):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False


    def initialize(self, problem, L, dim_x, dim_u):
        """
        Description: Initialize the dynamics of the model
        Args:
            dyn (function): dynamics of problem
            L (function): loss function
            dim_x (int): state_space dimension
            dim_u (int): action_space dimension
        """
        self.initialized = True

        # initialize dynamics, loss, and derivatives
        dyn = problem.dynamics
        self.dyn = dyn
        self.L = L
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dyn_jacobian = jax.jit(jax.jacrev(dyn, argnums=(0,1)))
        self.L_grad = jax.jit(jax.grad(L, argnums=(0,1)))
        self.L_hessian = jax.jit(jax.hessian(L, argnums=(0,1)))
        self.total_cost = jax.jit(lambda x, u: np.sum([self.L(x_t, u_t) for x_t, u_t in zip(x, u)])) # computes total cost over trajectory

        """ 
        Description: run LQR on provided matrices (this version computes delta-x and delta-u).
        dyn: dynamics, T: number of time steps to plan, x: current states, u: current actions
        F, f: dynamics linearization, C,c: cost linearization
        """
        @jax.jit
        def lqr_iteration(F_t, C_t, c_t, V_t, v_t, lamb):
            dim_x, dim_u = self.dim_x, self.dim_u

            Q = C_t + F_t.T @ V_t @ F_t
            q = c_t + F_t.T @ v_t

            q_x, q_u = q[:dim_x], q[dim_x:]
            Q_xx, Q_ux, Q_uu = Q[:dim_x, :dim_x], Q[dim_x:, :dim_x], Q[dim_x:, dim_x:]
            Q_uu_evals, Q_uu_evecs = np.linalg.eigh(Q_uu)
            Q_uu_evals = lamb + np.maximum(Q_uu_evals, 0)
            Q_uu_inv = Q_uu_evecs @ np.diag(1. / Q_uu_evals) @ Q_uu_evecs.T

            K_t = -Q_uu_inv @ Q_ux # shape (dim_u, dim_x)
            k_t = -Q_uu_inv @ q_u # shape (dim_u,)
            V_t = Q_xx + (Q_ux @ K_t.T + K_t.T @ Q_ux) + K_t.T @ Q_uu @ K_t
            v_t = q_x + Q_ux.T @ k_t + K_t.T @ q_u + K_t.T @ Q_uu @ k_t
            return K_t, k_t, V_t, v_t

        def lqr(T, x, u, F, C, c, lamb):
            def to_ndarray(x):
                x = np.asarray(x)
                if(np.ndim(x) == 0):
                    x = x[None, None]
                return x
            self.extend = lambda x, T: [to_ndarray(x) for i in range(T)]

            dim_x, dim_u = self.dim_x, self.dim_u

            u = self.extend(np.zeros((dim_u, )), T)
            K = self.extend(np.zeros((dim_u, dim_x)), T)
            k = u.copy()
            V = np.zeros((dim_x, dim_x))
            v = np.zeros((dim_x, ))
            Q = np.zeros((dim_x + dim_u, dim_x + dim_u))
            q = np.zeros((dim_x + dim_u, ))
            """
            K = T * [None]
            k = T * [None]
            V = np.zeros((dim_x, dim_x))
            v = np.zeros((dim_x, ))
            Q = np.zeros((dim_x + dim_u, dim_x + dim_u))
            q = np.zeros((dim_x + dim_u, ))
            """

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

                """
                if t > T-5:
                    print("t: " + str(t))
                    print("K: " + str(K[t]))
                    print("k: " + str(k[t]))
                    print("V: " + str(V))
                    print("v: " + str(v))
                    print("Q: " + str(Q))
                    print("Q_uu_inv: " + str(Q_uu_inv))
                    print("q: " + str(q))
                """

            ## Forward Recursion ##
            x_new = [x[0]]
            u_new = [0 for i in range(T)]
            for t in range(T):
                u_new[t] = u[t] + k[t] + K[t] @ (x_new[t] - x[t])            
                if t < T-1:
                    x_new.append(self.dyn(x_new[t], u_new[t]))

            return x_new, u_new

            """
            dim_x, dim_u = self.dim_x, self.dim_u
            V_t, v_t = np.zeros((dim_x, dim_x)), np.zeros((dim_x,))
            K, k = T * [None], T * [None]

            for t in reversed(range(T)): # backward pass
                K_t, k_t, V_t, v_t = lqr_iteration(F[t], C[t], c[t], V_t, v_t, lamb)
                K[t] = K_t
                k[t] = k_t

            x_stack, u_stack = [], []
            x_t = x[0]
            for t in range(T): # forward pass
                u_t = u[t] + K[t] @ (x_t - x[t]) + k[t] # d_x_t = x_t - x[t] # maybe we should just ignore k[t]?
                x_stack.append(x_t)
                u_stack.append(u_t)
                x_t = dyn(x_t, u_t)
            return x_stack, u_stack
            """
        self._lqr = lqr

        """ 
        Description: linearize provided system dynamics and loss, given initial state and actions. 
        """
        @jax.jit
        def linearization_iteration(x_t, u_t):
            block = lambda A: np.vstack([np.hstack([A[0][0], A[0][1]]), np.hstack([A[1][0], A[1][1]])]) # np.block not yet implemented
            F_t = np.hstack(dyn_jacobian(x_t, u_t))
            C_t = block(L_hessian(x_t, u_t))
            c_t = np.hstack(L_grad(x_t, u_t))
            return F_t, C_t, c_t

        def linearization(T, x, u):
            """
            F, C, c = [], [], [] # list appending is faster than matrix index update
            for t in range(T):
                F_t, C_t, c_t = linearization_iteration(x[t], u[t])
                F.append(F_t)
                C.append(C_t)
                c.append(c_t)
            return F, C, c
            """
            F = [np.hstack(dyn_jacobian(x[t],u[t])) for t in range(T)]
            C = [L_hessian(x[t],u[t]) for t in range(T)]
            C = [np.vstack([np.hstack([C_x[0],C_x[1]]), np.hstack([C_u[0],C_u[1]])]) for (C_x, C_u) in C]
            c = [L_grad(x[t],u[t]) for t in range(T)]
            c = [np.hstack([c_t[0], c_t[1]]) for c_t in c]
            return F, C, c
        self._linearization = linearization


    def ilqr(self, x_0, T, threshold=0.1, lamb=0.1, max_iterations=50):
        dim_x, dim_u = self.dim_x, self.dim_u
        x = [x_0]
        u = [np.zeros((dim_u,)) for t in range(T)]
        x_t = x_0
        for u_t in u:
            x.append(x_t)
            x_t = self.dyn(x_t, u_t)

        old_cost = self.total_cost(x, u)
        count = 0
        while count < max_iterations:
            count += 1
            if count > 10: break
            print("\ncount = " + str(count))
        
            #F, C, c = self._linearization(T, x, u)
            F = [np.hstack(self.dyn_jacobian(x[t],u[t])) for t in range(T)]
            C = [self.L_hessian(x[t],u[t]) for t in range(T)]
            C = [np.vstack([np.hstack([C_x[0],C_x[1]]), np.hstack([C_u[0],C_u[1]])]) for (C_x, C_u) in C]
            c = [self.L_grad(x[t],u[t]) for t in range(T)]
            c = [np.hstack([c_t[0], c_t[1]]) for c_t in c]

            x_new, u_new = self._lqr(T, x, u, F, C, c, lamb)

            new_cost = self.total_cost(x_new, u_new)
            if new_cost < old_cost:
                old_cost = new_cost
                x, u = x_new, u_new
                lamb *= 2.0 # this is the opposite of the regular iLQR algorithm, but seems to work much better
                #if np.abs(new_cost - old_cost) / old_cost < threshold:
                #    break
            else:
                lamb /= 2.0

        return u


    def predict(self, x_0, T, threshold=0.1, lamb=0.1, max_iterations=10):
        """
        Description: Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """
        return self.ilqr(x_0, T, threshold, lamb, max_iterations)

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
        return "<iLQR Model>"


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