"""
Test for PyBullet cartpole environment
"""
import time
import tigercontrol
import jax.numpy as np
from tigercontrol.environments.cartpole import CartPole
from tigercontrol.tasks.cartpole_v0 import CartPole_v0
from tigercontrol.controllers.ilqr import ILQR

# cartpole test
def test_cartpole(verbose=False):
    environment = tigercontrol.environment("CartPole")
    # environment = CartPole()
    # task = CartPole_v0()

    C_x, C_u = np.diag(np.array([0.1, 0.0, 1.0, 0.0])), np.diag(np.array([0.1]))
    L = lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u
    dim_x, dim_u = 4, 1
    update_period = 75
    obs = environment.reset()

    T = 75 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 25

    controller = ILQR(environment, max_iterations, lamb, threshold)
    # controller.initialize

    if verbose:
        print("Running iLQR...")
    u = controller.plan(obs, T)

    index = 0
    for t in range(10 * T):
        if verbose: 
            environment.render()
            time.sleep(1. / 50.)
        obs = environment.step(u[index])
        index += 1
        '''
        if done:
            if verbose:
                print("lasted {} time steps".format(t+1))
            obs = environment.reset()'''

        if index == T:
            if verbose:
                print("recomputing u...")
            obs = environment.reset()
            u = controller.plan(obs, T)
            # print("u : " + str(u))
            index = 0

    environment.close()
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
