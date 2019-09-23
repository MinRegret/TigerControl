"""
Test for PyBullet pendulum problem
"""
import time
import tigercontrol
import jax.numpy as np


# cartpole test
def test_pendulum(verbose=False):
    problem = tigercontrol.problem("Pendulum-v0")
    #L = lambda x, u: 1. - x[1] # 1 - cos(theta), where theta=0 is the goal (pendulum pointing up)
    L = lambda x, u: x[0]**2
    dim_x, dim_u = 2, 1
    obs = problem.initialize()

    model = tigercontrol.model("ILQR")
    model.initialize(problem, L, dim_x, dim_u)
    T = 100 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 50

    if verbose:
        print("Running iLQR...")
    u = model.plan(obs, T, max_iterations, lamb, threshold)

    #print("u: " + str([float(u_t) for u_t in u]))
    index = 0
    for t in range(25 * T):
        if verbose: 
            problem.render()

        time.sleep(1. / 30.)
        obs, r, done, _ = problem.step(u[index])
        index += 1

        if done:
            print("lasted {} time steps".format(t+1))
            obs = problem.initialize()
        if done or index == T:
            print("recomputing u...")
            u = model.plan(obs, T, max_iterations, lamb, threshold)
            index = 0

    problem.close()
    print("test_pendulum passed")


if __name__ == "__main__":
    test_pendulum(verbose=True)
