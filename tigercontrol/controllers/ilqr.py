"""
Iterative Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller

class ILQR(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    class Baby_controller(Controller):
        def __init__(self, u_old, x_old, K, k):
            self.u_old = u_old
            self.x_old = x_old
            self.K = K
            self.k = k
            self.t = 0
        def get_action(self, x):
            assert(self.t < len(self.x_old))
            '''
            print("self.u_old: " + str(self.u_old))
            print("self.x_old: " + str(self.x_old))
            print("self.K: " + str(self.K))
            print("self.k: " + str(self.k))
            print("self.t " + str(self.t))'''
            '''
            print("*********************************************************")
            print("self.t: " + str(self.t))
            print("self.K[self.t].shape: " + str(self.K[self.t].shape))
            print("type(x) : " + str(type(x)))
            print("type(self.x_old[self.t]) : " + str(type(self.x_old[self.t])))
            print("x: " + str(x))
            print("self.x_old[self.t] : " + str(self.x_old[self.t]))
            print("(x - self.x_old[self.t]).shape: " + str((x - self.x_old[self.t]).shape))'''
            u_next = self.u_old[self.t] + self.K[self.t] @ (x - self.x_old[self.t]) + self.k[self.t]
            self.t += 1
            return u_next

    compatibles = set([])

    def __init__(self):
        self.initialized = False


    def initialize(self, env, dim_x, dim_u, max_iterations, lamb, threshold, loss=None):
        """
        Description: Initialize the dynamics of the method
        Args:
            problem (instance/function): problem instance *OR* dynamics of problem
            L (function): loss function
            dim_x (int): state_space dimension
            dim_u (int): action_space dimension
        """
        self.initialized = True

        # self.env = env.
        # initialize dynamics, loss, and derivatives
        '''
        if callable(problem_dynamics):
            dyn = problem_dynamics
        else:
            dyn = problem_dynamics.dynamics
        self.dyn = dyn'''
        # self.L = env.loss
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.env = env
        # dyn_jacobian = jax.jit(jax.jacrev(dyn, argnums=(0,1)))
        # L_grad = jax.jit(jax.grad(self.L, argnums=(0,1)))
        # L_hessian = jax.jit(jax.hessian(self.L, argnums=(0,1)))
        self.L = self.env.get_loss()
        self.total_cost = jax.jit(lambda x, u: np.sum([self.L(x_t, u_t) for x_t, u_t in zip(x, u)])) # computes total cost over trajectory
        
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
        self._lqr_iteration = lqr_iteration

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

    def _form_next_controller(self, transcript):
        print("+++++++++++++++++++++++++++++++++++")
        print("transcript: " + str(transcript.keys()))
        T = len(transcript['x'])
        x_old = transcript['x']
        u_old = transcript['u']
        F = transcript['dynamics_grad']
        C = transcript['loss_hessian']
        c = transcript['loss_grad']
        lamb = self.lamb

        dim_x, dim_u = self.dim_x, self.dim_u
        V, v = np.zeros((dim_x, dim_x)), np.zeros((dim_x,))
        K, k = T * [None], T * [None]

        ## Backward Recursion ##
        for t in reversed(range(T)):
            K_t, k_t, V, v = self._lqr_iteration(F[t], C[t], c[t], V, v, lamb)
            K[t] = K_t
            k[t] = k_t
        return self.Baby_controller(u_old, x_old, K, k)

    def plan(self, x_0, T):
        dim_x, dim_u = self.dim_x, self.dim_u
        u_old = [np.zeros((dim_u,)) for t in range(T)]
        # x_0 = np.array([x_0[t] for t in range(len(x_0))])
        x_old = [x_0 for t in range(T)]
        # K, k = T * [np.zeros((dim_u, dim_x))], T * [np.zeros((dim_u,))]
        K, k = T * [np.zeros((dim_u, dim_x))], T * [np.zeros((dim_u,))]
        controller = self.Baby_controller(u_old, x_old, K, k)
        '''
        print("=====================================")
        print("dim_x = " + str(dim_x))
        print("dim_u = " + str(dim_u))
        print("type(x_0) = " + str(type(x_0)))
        print("x_0.shape = " + str(x_0.shape))
        print("x_0.T.shape = " + str(x_0.T.shape))
        print("x_old[0].shape = " + str(x_old[0].shape))
        print("u_old[0].shape = " + str(u_old[0].shape))'''
        old_cost = self.total_cost(x_old, u_old)
        count = 0
        transcript_old = {'x' : x_old, 'u' : u_old}

        while count < self.max_iterations:
            count += 1
            transcript = self.env.rollout(controller, T, dynamics_grad=True, loss_grad=True, loss_hessian=True)
            x_new, u_new = transcript['x'], transcript['u']
            
            new_cost = self.total_cost(x_new, u_new)
            print("=================================")
            print("old_cost = " + str(old_cost))
            print("new_cost = " + str(new_cost))
            if new_cost < old_cost or count == 1:
                transcript_old = transcript
                if self.threshold and count > 1 and (old_cost - new_cost) / old_cost < self.threshold:
                    break
                self.lamb /= 2.0
                old_cost = new_cost
            else:
                transcript = transcript_old
                self.lamb *= 2.0

            controller = self._form_next_controller(transcript)
        return transcript['u']

    def update(self):
        pass

    def __str__(self):
        return "<iLQR Method>"
