import tigercontrol
import jax
import jax.numpy as np
from tigercontrol.controllers import Controller
from tigercontrol.utils.optimizers.losses import mse

quad = lambda x, u: np.sum(x.T @ x + u.T @ u)

class DynaBoost(Controller):
    """
    Description: 
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, A, B, controller_id, controller_hyperparams, N = 3, H = 3, cost_fn = quad):
        """
        Description: Initializes autoregressive controller parameters
        Args:
            controller_id (string): id of weak learner controller
            controller_params (dict): dict of params to pass controller
            N (int): default 3. Number of weak learners
            loss (function): loss function for boosting controller
            reg (float): default 1.0. constant for regularization.
        """
        self.initialized = True

        n, m = B.shape # State & Action Dimensions
        self.A, self.B = A, B # System Dynamics

        # 1. Maintain N copies of the algorithm 
        assert N > 0
        self.N = N
        self.controllers = []

        self.past_partial_actions = np.zeros((N, H, m, 1))

        self.x = np.zeros((n, 1))
        # Past H noises
        self.w = np.zeros((H, n, 1))

        # 2. Initialize the N weak learners
        for _ in range(N):
            new_controller = tigercontrol.controllers(controller_id)
            new_controller.initialize(**controller_hyperparams)
            self.controllers.append(new_controller)

        # 3. Define a proxy loss which only penalizes the last H actions
        def proxy_loss(actions, w, cost_t = cost_fn):
             """
            Description: Initializes autoregressive controller parameters
            Args:
                u: sequence of past H actions
            """
            y = np.zeros((n, 1))
            for h in range(H - 1):
                #y = dynamics(y, actions[h]) + w[h]
                y = A @ y + B @ actions[h] + w[h]

            return cost_t(y, actions[h]) 

        # 4. Extract the set of actions of previous learners
        def get_partial_actions(x):
            u = [np.zeros((m, 1))]
            partial_u = np.zeros((m, 1))
            for i, controller_i in enumerate(self.controllers):
                eta_i = 2 / (i + 2)
                pred_u = controller_i.get_action(x)
                partial_u = (1 - eta_i) * partial_u + eta_i * pred_u
                u.append(cur_u)
            return np.array(u)
        self.get_partial_actions = get_partial_actions

        # 5. Extract the set of actions of previous learners
        def get_grads(partial_actions, w, cost_t = cost_fn):
            v_list = [jax.grad(proxy_loss(partial_actions[i], w, cost_t)) for i in range(N)]
            return v_list

        self.get_grads = jax.jit(get_grads)


    def get_action(self, x):

        # 1. Get partial action
        partial_actions = self.get_partial_actions(x)


        # N x H 
        self.past_partial_actions = jax.ops.index_update(self.past_partial_actions, \
                                    0, x - self.A @ self.x - self.B @ self.u)
        self.past_partial_actions = np.roll(self.past_partial_actions, -1, axis = 0)

        # 1. Get new noise (will be located at w[-1])
        self.w = jax.ops.index_update(self.w, 0, x - self.A @ self.x - self.B @ self.u)
        self.w = np.roll(self.w, -1, axis = 0)

        # 2. Update x
        self.x = x

        return self.past_partial_actions[-1]


    def update(self, cost = None):
        grads = self.get_grads(self.past_partial_actions, self.w, cost)
        for i, controller_i, grad_i in zip(enumerate(self.controllers), grads[:-1]):
            ### WHAT IF PARAMS isn't the correct shape?? 
            ## e.g. what about bias?
            ## can I set the boosting H to be number of parameters
            linear_loss_i = np.dot(grad_i, self.controller_actions)
            controller_i.update(grad = jax.grad(linear_loss_i))



