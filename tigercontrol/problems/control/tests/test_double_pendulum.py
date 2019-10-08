"""
Test for PyBullet double pendulum problem
"""
import time
import tigercontrol
import jax.numpy as np


# double pendulum test #TODO: finish
def test_double_pendulum(verbose=False):
    problem = tigercontrol.problem("DoublePendulum-v0")
    # observe [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
    L = lambda x, u: x[0] - x[2]
    dim_x, dim_u = 4, 1
    obs = problem.initialize()

    method = tigercontrol.method("ILQR")
    method.initialize(problem, L, dim_x, dim_u)
    T = 30 # horizon
    threshold = None
    lamb = 0.1
    max_iterations = 25

    if verbose:
        print("Running iLQR...")
    u = method.plan(obs, T, max_iterations, lamb, threshold)

    index = 0
    for t in range(5 * T):
        if verbose: 
            problem.render()
            time.sleep(1. / 15.)
        obs, r, done, _ = problem.step(u[index])
        index += 1

        if done:
            if verbose:
                print("solved double pendulum in {} time steps!".format(t+1))
            obs = problem.initialize()
        if done or index == T:
            if verbose:
                print("recomputing u...")
            u = method.plan(obs, T, max_iterations, lamb, threshold)
            index = 0

    problem.close()
    print("test_double_pendulum passed")


if __name__ == "__main__":
    test_double_pendulum(verbose=True)
