import tigercontrol
from tigercontrol.controllers import Controller
from tigercontrol.controllers.core import quad_loss, action_loss
from tigercontrol.controllers.core import update_noise

import jax
import jax.numpy as np
from jax import grad

class DynaBoost(Controller):
    """
    Description: 
    """

    def __init__(self, env, controller_id, controller_hyperparams, N = 3, H = 3, cost_fn = quad_loss):
        """
        Description: Initializes autoregressive controller parameters
        Args:
            controller_id (string): id of weak learner controller
            controller_params (dict): dict of params to pass controller
            N (int): default 3. Number of weak learners
        """
        self.initialized = True

        self.n, self.m = env.n, env.m # State & Action Dimensions
        self.env = env # System

        # 1. Maintain N copies of the algorithm 
        assert N > 0
        self.N, self.H = N, H
        self.controllers = []

        #past state
        self.x = np.zeros((self.n, 1))
        # Past 2H noises
        self.w = np.zeros((2 * H, self.n, 1))

        # 2. Initialize the N weak learners
        weak_controller_class = tigercontrol.controllers(controller_id)
        self.weak_controller = controller_class(controller_hyperparams)
        for _ in range(N):
            new_controller = new_controller_class(controller_hyperparams)
            self.controllers.append(new_controller)

        self.past_partial_actions = np.zeros((N+1, H, self.m, 1))

        # Extract the set of actions of previous learners
        def get_partial_actions(x):
            u = np.zeros((self.N + 1, self.m, 1))
            partial_u = np.zeros((self.m, 1))
            for i, controller_i in enumerate(self.controllers):
                eta_i = 2 / (i + 2)
                pred_u = controller_i.get_action(x)
                partial_u = (1 - eta_i) * partial_u + eta_i * pred_u
                u = jax.ops.index_update(u, i + 1, partial_u)
            return u

        self.get_partial_actions = get_partial_actions

        self.grad_action = grad(action_loss)

        # Extract the set of actions of previous learners
        def get_grads(partial_actions, w, cost_fn = quad_loss):
            v_list = [self.grad_action(partial_actions[i], w, self.H, self.env, cost_fn) for i in range(self.N)]
            return v_list

        self.get_grads = get_grads

        def linear_loss(controller_i_params, grad_i, w):
            linear_loss_i = 0

            y = np.zeros((n, 1))

            for h in range(self.H):
                v = self.weak_controller.determine_action(controller_i_params, y, w[:h+H])
                linear_loss_i += np.dot(grad_i[h], v)
                y = self.env.dyn(y, v) + w[h+H]

            v = self.weak_controller.determine_action(controller_i_params, y, w[:h+H])
            linear_loss_i += np.dot(grad_i[h], v)

            return np.sum(linear_loss_i)

        self.grad_linear = grad(linear_loss)

    def update_params(self, cost_fn = quad_loss, cost_val = None):

        grads = self.get_grads(self.past_partial_actions, self.w, cost_fn)

        for controller_i, grad_i in zip(self.controllers, grads):
            controller_i.update_params(grad = self.grad_linear(controller_i.params, grad_i, self.w))

    def update_history(self, x = None):
        self.w = update_noise(self.w, x, self.past_partial_actions[-1][-1], self.env)
        self.x = x
        self.past_partial_actions = jax.ops.index_update(self.past_partial_actions,\
                                    jax.ops.index[:, 0], self.get_partial_actions(x))
        self.past_partial_actions = np.roll(self.past_partial_actions, -1, axis = 1)

    def get_action(self, x):
        return self.get_partial_actions(x)[-1]


