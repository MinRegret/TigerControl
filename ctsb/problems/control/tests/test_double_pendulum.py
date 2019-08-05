"""
Test for PyBullet double pendulum problem
"""
import time
import ctsb
import jax.numpy as np


# double pendulum test #TODO: finish
def test_cartpole(verbose=False):
    problem = ctsb.problem("DoublePendulum-v0")
    C_x, C_u = np.diag(np.array([0.1, 0.0, 1.0, 0.0])), np.diag(np.array([0.1]))
    L = lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u
    dim_x, dim_u = 4, 1
    obs = problem.initialize()

    model = ctsb.model("ILQR")
    model.initialize(problem, L, dim_x, dim_u)
    T = 100 # horizon
    threshold = 0.05
    lamb = 0.1
    max_iterations = 25

    if verbose:
        print("Running iLQR...")
    u = model.ilqr(obs, T, threshold, lamb, max_iterations)

    index = 0
    for t in range(5 * T):
        if verbose: 
            problem.render()

        time.sleep(1. / 50.)
        obs, r, done, _ = problem.step(u[index])
        index += 1

        if done:
            print("lasted {} time steps".format(t+1))
            obs = problem.initialize()
        if done or index == T:
            print("recomputing u...")
            u = model.ilqr(obs, T, threshold, lamb, max_iterations)
            index = 0

    problem.close()
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
