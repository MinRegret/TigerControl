"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np


# cartpole test
def test_cartpole(verbose=False):
    problem = ctsb.problem("CartPole-v0")
    L = lambda x, u: x[2]**2
    dim_x, dim_u = 4, 1
    obs = problem.initialize()

    model = ctsb.model("ILQR")
    model.initialize(problem, L, dim_x, dim_u)
    H = 100 # horizon
    threshold = 0.1
    lamb = 1.0
    max_iterations = 10

    if verbose:
        print("Running iLQR...")
    T = 1000
    u = model.ilqr(obs, H, threshold, lamb, max_iterations)
    cur_index = 0
    for t in range(T):
        if verbose: 
            problem.render()

        if cur_index == H:
            u = model.ilqr(obs, H, threshold, lamb, max_iterations)
            cur_index = 0

        time.sleep(1. / 60.)
        obs, r, done, _ = problem.step(u[cur_index])

        if done:
            break

            # alternatively, continue
            print("done! problem resetting...")
            obs = problem.initialize()
            u = model.ilqr(obs, H, threshold, lamb, max_iterations)
            cur_index = 0

    problem.close()
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
