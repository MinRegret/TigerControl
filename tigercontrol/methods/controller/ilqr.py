"""
Iterative Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import tigercontrol
from tigercontrol.methods import Method


class ILQR(Method):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False


    def initialize(self, environment_dynamics, L, dim_x, dim_u, update_period, max_iterations, lamb, threshold):
        """
        Description: Initialize the dynamics of the method
        Args:
            environment (instance/function): environment instance *OR* dynamics of environment
            L (function): loss function
            dim_x (int): state_space dimension
            dim_u (int): action_space dimension
        """
        self.initialized = True

        # initialize dynamics, loss, and derivatives
        if callable(environment_dynamics):
            dyn = environment_dynamics
        else:
            dyn = environment_dynamics.dynamics
        self.dyn = dyn
        self.L = L
        self.dim_x = dim_x
        self.dim_u = dim_u
        dyn_jacobian = jax.jit(jax.jacrev(dyn, argnums=(0,1)))
        L_grad = jax.jit(jax.grad(L, argnums=(0,1)))
        L_hessian = jax.jit(jax.hessian(L, argnums=(0,1)))
        self.total_cost = jax.jit(lambda x, u: np.sum([self.L(x_t, u_t) for x_t, u_t in zip(x, u)])) # computes total cost over trajectory
        
        self.t = 0 # time counter
        self.update_period = update_period # update when t % update_period == 0
        self.current_plan = [] # stores currently planned trajectory
        self.max_iterations = max_iterations
        self.lamb = lamb
        self.threshold = threshold

        """ 
        Description: run LQR on provided matrices (this version computes delta-x and delta-u).
        dyn: dynamics, T: number of time steps to plan, x: current states, u: current actions
        F, f: dynamics linearization, C,c: cost linearization
        """
        @jax.jit
        def lqr_iteration(F_t, C_t, c_t, V, v, lamb):
            dim_x, dim_u = self.dim_x, self.dim_u
            Q = C_t + F_t.T @ V @ F_t
            q = c_t + F_t.T @ v

            Q_uu, Q_ux, Q_xx = Q[dim_x:, dim_x:], Q[dim_x:, :dim_x], Q[:dim_x, :dim_x]
            q_u, q_x = q[dim_x:], q[:dim_x]
            Q_uu_evals, Q_uu_evecs = np.linalg.eigh(Q_uu)
            Q_uu_evals = lamb + np.maximum(Q_uu_evals, 0.0)
            Q_uu_inv = Q_uu_evecs @ np.diag(1. / Q_uu_evals) @ Q_uu_evecs.T

            K_t = -Q_uu_inv @ Q_ux
            k_t = -Q_uu_inv @ q_u
            V = Q_xx + Q_ux.T @ K_t + K_t.T @ Q_ux + K_t.T @ Q_uu @ K_t
            v = q_x + Q_ux.T @ k_t + K_t.T @ q_u + K_t.T @ Q_uu @ k_t
            return K_t, k_t, V, v

        def lqr(T, x, u, F, C, c, lamb):
            dim_x, dim_u = self.dim_x, self.dim_u
            V, v = np.zeros((dim_x, dim_x)), np.zeros((dim_x,))
            K, k = T * [None], T * [None]

            ## Backward Recursion ##
            for t in reversed(range(T)):
                K_t, k_t, V, v = lqr_iteration(F[t], C[t], c[t], V, v, lamb)
                K[t] = K_t
                k[t] = k_t

            ## Forward Recursion ##
            x_stack, u_stack = [], []
            x_t = x[0]
            for t in range(T):
                u_t = u[t] + K[t] @ (x_t - x[t]) + k[t]
                x_stack.append(x_t)
                u_stack.append(u_t)
                x_t = dyn(x_t, u_t)
            return x_stack, u_stack
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
            F, C, c = [], [], [] # list appending is faster than matrix index update
            for t in range(T):
                F_t, C_t, c_t = linearization_iteration(x[t], u[t])
                F.append(F_t)
                C.append(C_t)
                c.append(c_t)
            return F, C, c
        self._linearization = linearization


    def plan(self, x):
        if self.t % self.update_period == 0:
            self.current_plan = self.plan_trajectory(x, self.update_period, max_iterations=self.max_iterations, lamb=self.lamb, threshold=self.threshold)
        next_u = self.current_plan[self.t % self.update_period]
        self.t += 1
        return next_u

    def plan_trajectory(self, x_0, T, max_iterations=10, lamb=0.1, threshold=None):
        dim_x, dim_u = self.dim_x, self.dim_u
        u = [np.zeros((dim_u,)) for t in range(T)]
        x = [x_0]
        [x.append(self.dyn(x[-1], u_t)) for u_t in u]

        old_cost = self.total_cost(x, u)
        count = 0
        while count < max_iterations:
            count += 1
            F, C, c = self._linearization(T, x, u)
            x_new, u_new = self._lqr(T, x, u, F, C, c, lamb)

            new_cost = self.total_cost(x_new, u_new)
            if new_cost < old_cost:
                x, u = x_new, u_new
                if threshold and (old_cost - new_cost) / old_cost < threshold:
                    break
                lamb /= 2.0
                old_cost = new_cost
            else:
                lamb *= 2.0
        return u

    def __str__(self):
        return "<iLQR Method>"

