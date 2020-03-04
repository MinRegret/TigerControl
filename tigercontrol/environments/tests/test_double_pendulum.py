"""
Test for PyBullet double pendulum environment
"""
import time
import tigercontrol
import jax.numpy as np


# double pendulum test #TODO: finish
def test_double_pendulum(verbose=False):
    environment = tigercontrol.environment("DoublePendulum")
    # observe [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
    L = lambda x, u: (x[0] - x[2])**2
    dim_x, dim_u = 4, 1
    obs = environment.initialize()

    update_period = 75
    T = 75 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 25

    controller = tigercontrol.controllers("ILQR")
    controller.initialize(environment, L, dim_x, dim_u, update_period, max_iterations, lamb, threshold)

    if verbose:
        print("Running iLQR...")
    # u = controller.plan(obs, T, max_iterations, lamb, threshold)

    index = 0
    for t in range(10 * T):
        if verbose: 
            environment.render()
            time.sleep(1. / 15.)
        u = controller.plan(obs)
        obs, r, done, _ = environment.step(u)
        index += 1

        if done:
            if verbose:
                print("solved double pendulum in {} time steps!".format(t+1))
            obs = environment.initialize()
        '''
        if done or index == T:
            if verbose:
                print("recomputing u...")
            u = controller.plan(obs, T, max_iterations, lamb, threshold)
            index = 0'''

    environment.close()
    print("test_double_pendulum passed")


if __name__ == "__main__":
    test_double_pendulum(verbose=True)
