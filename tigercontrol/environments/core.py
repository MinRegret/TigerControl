# Environment class
# Author: John Hallman

from tigercontrol import error
import inspect
import jax
import jax.numpy as np

# class for online control tests
class Environment(object):
    def __init__(self):
        self.initialized = False
        self.n = None
        self.m = None
        self._dynamics = None
        self._state = None
        self._dynamics_jacobian = None
        self._loss = None
        self._loss_grad = None
        self._loss_hessian = None
    def reset(self):
        ''' Description: reset environment to state at time 0, return state. '''
        raise NotImplementedError

    def step(self, **kwargs):
        ''' Description: run one timestep of the environment's dynamics. '''
        raise NotImplementedError


    def get_state_dim(self):
        ''' Description: return dimension of the state. '''
        return self.n
        # raise NotImplementedError

    def get_action_dim(self):
        ''' Description: return dimension of action inputs. '''
        return self.m
        # raise NotImplementedError

    def get_state_dim(self):
        return self.n

    def get_action_dim(self):
        return self.m

    def get_dynamics(self):
        return self._dynamics

    def get_state(self):
        return self._state

    def get_dynamics_jacobian(self):
        return self._dynamics_jacobian

    def get_loss(self):
        return self._loss

    def get_loss_grad(self):
        return self._loss_grad

    def get_loss_hessian(self):
        return self._loss_hessian

    def rollout(self, baby_controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        """ Description: Roll out and return trajectory of given baby_controller. """
        raise NotImplementedError

    def get_loss(self):
        return self._loss

    def get_terminal_loss(self):
        return self._terminal_loss

    def close(self):
        ''' Description: closes the environment and returns used memory '''
        pass

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __repr__(self):
        return self.__str__()
        
""" # OLD CODE
    def rollout(self, baby_controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        request_grad = dynamics_grad or loss_grad or loss_hessian
        if not hasattr(self, "compiled") and request_grad: # on first call, compile gradients
            if '_dynamics' not in vars(self):
                raise NotImplementedError("rollout not possible: no dynamics in {}".format(self))
            if '_loss' not in vars(self):
                raise NotImplementedError("rollout not possible: no loss in {}".format(self))
            try:
                # stack the jacobians of environment dynamics gradient
                jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
                self._dynamics_jacobian = jax.jit(lambda x, u: np.hstack(jacobian(x, u)))
                # stack the gradients of environment loss
                loss_grad = jax.grad(self._loss, argnums=(0,1))
                self._loss_grad = jax.jit(lambda x, u: np.hstack(loss_grad(x, u)))
                # block the hessian of environment loss
                block_hessian = lambda A: np.vstack([np.hstack([A[0][0], A[0][1]]), np.hstack([A[1][0], A[1][1]])])
                hessian = jax.hessian(self._loss, argnums=(0,1))
                self._loss_hessian = jax.jit(lambda x, u: block_hessian(hessian(x,u)))
            except Exception as e:
                print(e)
                raise error.JAXCompilationError("jax.jit failed to compile environment dynamics or loss")
            self.compiled = True

        transcript = {'x': [], 'u': []} # return transcript
        if dynamics_grad: transcript['dynamics_grad'] = [] # optional derivatives
        if loss_grad: transcript['loss_grad'] = []
        if loss_hessian: transcript['loss_hessian'] = []

        x_origin, x = self._state, self._state
        for t in range(T):
            u = baby_controller.get_action(x)
            transcript['x'].append(x)
            transcript['u'].append(u)
            if dynamics_grad: transcript['dynamics_grad'].append(self._dynamics_jacobian(x, u))
            if loss_grad: transcript['loss_grad'].append(self._loss_grad(x, u))
            if loss_hessian: transcript['loss_hessian'].append(self._loss_hessian(x, u))
            x = self.step(u)[0] # move to next state
        self._state = x_origin # return to original state
        return transcript
    """

