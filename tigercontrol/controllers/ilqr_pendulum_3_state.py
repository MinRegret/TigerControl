"""
Iterative Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import jax.random as random

import tigercontrol
from tigercontrol.controllers import Controller
from tigercontrol.utils import generate_key, get_tigercontrol_dir

class ILQR(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    class OpenLoopController(Controller):
        def __init__(self, u_old, x_old, K, k, alpha):
            self.u_old = u_old
            self.x_old = x_old
            self.K = K
            self.k = k
            self.t = 0
            self.alpha = alpha
        def get_action(self, x):
            assert(self.t < len(self.x_old))
            u_next = self.u_old[self.t] + self.K[self.t] @ (x - self.x_old[self.t]) + self.alpha * self.k[self.t]
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
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.env = env
        self.L = self.env.get_loss()
        self.terminal_L = self.env.get_terminal_loss()
        self.total_cost = jax.jit(lambda x, u: np.sum([self.L(x_t, u_t) for x_t, u_t in zip(x[:-1], u)]) + self.terminal_L(x[-1], [0.0])) # computes total cost over trajectory
        
        self.max_iterations = max_iterations
        self.lamb = lamb
        self.threshold = threshold
        self.K_alpha1 = None
        self.k_alpha1 = None

        """ 
        Description: run LQR on provided matrices (this version computes delta-x and delta-u).
        dyn: dynamics, T: number of time steps to plan, x: current states, u: current actions
        F, f: dynamics linearization, C,c: cost linearization
        """
        @jax.jit
        def lqr_iteration(F_t, C_t, c_t, V, v, lamb):
            dim_x, dim_u = self.dim_x, self.dim_u

            reg = lamb * np.eye(dim_x) # modified reg

            Q_reg = C_t + F_t.T @ (V + reg) @ F_t # modified reg
            Q = C_t + F_t.T @ V @ F_t
            q = c_t + F_t.T @ v

            Q_uu, Q_ux = Q_reg[dim_x:, dim_x:], Q_reg[dim_x:, :dim_x] # modified reg
            Q_xx = Q[:dim_x, :dim_x]
            q_u, q_x = q[dim_x:], q[:dim_x]
            
            '''
            Q_uu_evals, Q_uu_evecs = np.linalg.eigh(Q_uu)
            Q_uu_evals = lamb + np.maximum(Q_uu_evals, 0.0)
            Q_uu_inv = Q_uu_evecs @ np.diag(1. / Q_uu_evals) @ Q_uu_evecs.T'''

            K_t = -np.linalg.solve(Q_uu, Q_ux) # modified reg
            k_t = -np.linalg.solve(Q_uu, q_u) # modified reg
            # K_t = -Q_uu_inv @ Q_ux
            # k_t = -Q_uu_inv @ q_u
            V = Q_xx + Q_ux.T @ K_t + K_t.T @ Q_ux + K_t.T @ Q_uu @ K_t
            v = q_x + Q_ux.T @ k_t + K_t.T @ q_u + K_t.T @ Q_uu @ k_t


            V = 0.5 * (V + V.T) # modified for symmetry
            return K_t, k_t, V, v
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

    def _form_next_controller(self, transcript, alpha):
        T = len(transcript['x'])-1
        x_old = transcript['x']
        u_old = transcript['u']

        if alpha == 1:
            F = transcript['dynamics_grad']
            C = transcript['loss_hessian']
            c = transcript['loss_grad']
            lamb = self.lamb

            dim_x, dim_u = self.dim_x, self.dim_u
            # V, v = np.zeros((dim_x, dim_x)), np.zeros((dim_x,))
            V, v = C[-1][:dim_x, :dim_x], c[-1][:dim_x] # modified initial V, v
            K, k = T * [None], T * [None]

            ## Backward Recursion ##
            for t in reversed(range(T)):
                K_t, k_t, V, v = self._lqr_iteration(F[t], C[t], c[t], V, v, lamb)
                K[t] = K_t
                k[t] = k_t
            self.K_alpha1 = K
            self.k_alpha1 = k
        return self.OpenLoopController(u_old, x_old, self.K_alpha1, self.k_alpha1, alpha)

    def plan(self, x_0, T):
        dim_x, dim_u = self.dim_x, self.dim_u
        u_old = random.uniform(generate_key(), shape=(T,dim_u), minval=self.env.min_bounds, maxval=self.env.max_bounds)
        # print(u_old)
        pos = 0
        for i in range(len(u_old)):
            if u_old[i][0] > 0:
                pos += 1
        u_0 = u_old
        # print("pos = " + str(pos))
        # print("neg = " + str(T-pos))
        # u_old = [np.zeros((dim_u,)) for t in range(T)]
        '''x_old = [x_0]
        for t in range(T):
            x_old.append(self.env.dynamics(x_old[-1], u_old[t]))'''
        x_old = [x_0 for t in range(T+1)]
        K, k = T * [np.zeros((dim_u, dim_x))], T * [np.zeros((dim_u,))]
        controller = self.OpenLoopController(u_old, x_old, K, k, 1.0)
        opt_cost = self.total_cost(x_old, u_old)
        count = 0
        transcript_opt = {'x' : x_old, 'u' : u_old}
        # print("lamb:" + str(self.lamb))
        delta = 2.0
        delta_0 = 8.0
        lamb_min = 1e-6
        lamb_max = 1e5

        alphas = 1.1**(-np.arange(10)**2)
        converged = False
        accepted = False

        while count < self.max_iterations:
            count += 1
            alpha_count = 0
            accepted = False
            for alpha in alphas:
                # print("alpha = " + str(alpha))
                alpha_count += 1
                transcript = self.env.rollout(controller, T, dynamics_grad=True, loss_grad=True, loss_hessian=True)
                x_new, u_new = transcript['x'], transcript['u']
                
                new_cost = self.total_cost(x_new, u_new)
                if alpha_count == 1 and count == 1:
                    transcript_opt = transcript
                    opt_cost = new_cost
                else:
                    if new_cost < opt_cost:
                        print("ACCEPTED")
                        print("opt_cost = " + str(opt_cost))
                        print("new_cost = " + str(new_cost))
                        
                        
                        accepted = True
                        if self.threshold and count > 1 and (opt_cost - new_cost) / opt_cost < self.threshold:
                            print("CONVERGED")
                            print("opt_cost = " + str(opt_cost))
                            print("new_cost = " + str(new_cost))
                            converged = True

                        transcript_opt = transcript
                        opt_cost = new_cost
                        # delta /= delta_0
                        self.lamb /= delta_0
                        if self.lamb < lamb_min:
                            self.lamb = 0
                        break
                    else:
                        transcript = transcript_opt
                controller = self._form_next_controller(transcript, alpha)
            if not accepted:
                print("NOT ACCEPTED")
                # delta *= delta_0
                self.lamb = max(lamb_min, self.lamb * delta_0)
                if self.lamb >= lamb_max:
                    break
            if converged:
                break
            print("---------------------------")
            print("opt_cost = " + str(opt_cost))
            print("lamb = " + str(self.lamb))
            print("final state: " + str(self.reduce_state(transcript_opt['x'][-1])))
        state_trajectory = [self.reduce_state(state) for state in transcript_opt['x']]
        # print("state_trajectory: ")
        # print(state_trajectory)
        return transcript_opt['u'], pos, u_0

    def update(self):
        pass

    def reduce_state(self, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.
        In this case, it converts:
            [sin(theta), cos(theta), theta'] -> [theta, theta']
        Args:
            state: Augmented state vector [state_size].
        Returns:
            Reduced state size [reducted_state_size].
        """
        sin_theta, cos_theta, theta_dot = state

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([theta, theta_dot])

    def __str__(self):
        return "<iLQR Method>"
