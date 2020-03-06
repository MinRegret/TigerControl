"""
Test for PyBullet pendulum environment
"""
import time
import tigercontrol
import jax.numpy as np


# cartpole test
def test_environment_api(verbose=False):
    environment_ids = tigercontrol.environment_registry.list_ids() # get all controllers

    for environment_id in environment_ids:
        environment_class = tigercontrol.environment(environment_id)
        env = environment_class()

        # test APIs
        n, m = env.get_state_dim(), env.get_action_dim() # get dimensions of system
        x_0 = env.reset() # first state
        x_1 = env.step(np.zeros(m)) # try step with 0 action
        assert x_0 = env.reset() # assert reset return back to original state


if __name__ == "__main__":
    test_environment_api(verbose=True)
