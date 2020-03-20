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
    
    class OpenLoopController(Controller):
        def __init__(self, u_old, x_old, K, k):
            self.u_old = u_old
            self.x_old = x_old
            self.K = K
            self.k = k
            self.t = 0
        def get_action(self, x):
            assert(self.t < len(self.x_old))
            u_next = self.u_old[self.t] + self.K[self.t] @ (x - self.x_old[self.t]) + self.k[self.t]
            self.t += 1
            y = self.K[self.t] @ (x - self.x_old[self.t])
            z = (x - self.x_old[self.t])
            return u_next

    compatibles = set([])

    def __init__(self, env, max_iterations, lamb, threshold, loss=None):
        """
        Description: Initialize the dynamics of the method
        Args:
            problem (instance/function): problem instance *OR* dynamics of problem
            L (function): loss function
        """
        self.env = env
        self.dim_x, self.dim_u = env.get_state_dim(), env.get_action_dim()
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

        def _rollout(act, dyn, x_0, T):
            def f(x, i):
                u = act(x)
                x_next = dyn(x, u)
                return x_next, np.hstack((x,u))
                # return np.squeeze(x_next, axis=1), np.hstack((x, u))
                # return x_next, np.vstack((x, u))
            _, trajectory = jax.lax.scan(f, x_0, np.arange(T))
            return trajectory
        self._rollout = jax.jit(_rollout, static_argnums=(0,1,3))

    def _form_next_controller(self, transcript):
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
        return self.OpenLoopController(u_old, x_old, K, k)

    def plan(self, x_0, T):
        dim_x, dim_u = self.dim_x, self.dim_u
        u_old = [np.zeros((dim_u,)) for t in range(T)]
        x_old = [np.reshape(x_0, (dim_x,)) for t in range(T)]
        K, k = T * [np.zeros((dim_u, dim_x))], T * [np.zeros((dim_u,))]
        controller = self.OpenLoopController(u_old, x_old, K, k)
        old_cost = self.total_cost(x_old, u_old)
        count = 0
        transcript_old = {'x' : x_old, 'u' : u_old}

        while count < self.max_iterations:
            count += 1
            transcript = self.rollout(controller, T, dynamics_grad=True, loss_grad=True, loss_hessian=True)
            x_new, u_new = transcript['x'], transcript['u']
            
            new_cost = self.total_cost(x_new, u_new)
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

    def rollout(self, controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        # if self.rollout_controller != controller: self.rollout_controller = controller
        x = self.env.get_state()
        x = np.squeeze(x, axis=1)
        trajectory = self._rollout(controller.get_action, self.env.get_dynamics(), x, T)
        transcript = {'x': trajectory[:,:self.dim_x], 'u': trajectory[:,self.dim_x:]}
        # transcript = {'x': trajectory[:,:self.dim_x,], 'u': trajectory[:,self.dim_x:,]}

        # optional derivatives
        if dynamics_grad: transcript['dynamics_grad'] = []
        if loss_grad: transcript['loss_grad'] = []
        if loss_hessian: transcript['loss_hessian'] = []
        for x, u in zip(transcript['x'], transcript['u']):
            if dynamics_grad: transcript['dynamics_grad'].append(self.env.get_dynamics_jacobian()(x, u))
            if loss_grad: transcript['loss_grad'].append(self.env.get_loss_grad()(x, u))
            if loss_hessian: transcript['loss_hessian'].append(self.env.get_loss_hessian()(x, u))
        # transcript['x'] = [np.reshape(x, (x.shape[0], 1)) for x in transcript['x']]
        # transcript['u'] = [np.reshape(u, (u.shape[0], 1)) for u in transcript['u']]
        return transcript

    def update(self):
        pass

    def __str__(self):
        return "<iLQR Method>"