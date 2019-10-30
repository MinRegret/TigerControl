"""
Test for PyBullet pendulum environment
"""
import time
import tigercontrol
import jax.numpy as np


# cartpole test
def test_pendulum(verbose=False):
    environment = tigercontrol.environment("Pendulum-v0")
    #L = lambda x, u: 1. - x[1] # 1 - cos(theta), where theta=0 is the goal (pendulum pointing up)
    L = lambda x, u: x[0]**2
    dim_x, dim_u = 2, 1
    obs = environment.initialize()

    
    T = 100 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 50
    update_period = 100

    controller = tigercontrol.controllers("ILQR")
    controller.initialize(environment, L, dim_x, dim_u, update_period, max_iterations, lamb, threshold)

    if verbose:
        print("Running iLQR...")
    # u = controller.plan(obs, T, max_iterations, lamb, threshold)

    #print("u: " + str([float(u_t) for u_t in u]))
    index = 0
    for t in range(25 * T):
        if verbose: 
            environment.render()
            time.sleep(1. / 30.)
        u = controller.plan(obs)
        obs, r, done, _ = environment.step(u)
        index += 1

        if done:
            if verbose:
                print("lasted {} time steps".format(t+1))
            obs = environment.initialize()
        '''
        if done or index == T:
            if verbose:
                print("recomputing u...")
            u = controller.plan(obs, T, max_iterations, lamb, threshold)
            index = 0'''

    environment.close()
    print("test_pendulum passed")


if __name__ == "__main__":
    test_pendulum(verbose=True)
