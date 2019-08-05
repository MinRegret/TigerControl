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
    threshold = 0.05
    lamb = 0.1
    max_iterations = 20

    if verbose:
        print("Running iLQR...")
    #T = 1000
    u = model.ilqr(obs, H, threshold, lamb, max_iterations)

    print("u: " + str([float(u_t) for u_t in u]))
    for t in range(H):
        if verbose: 
            problem.render()

        time.sleep(2. / 60.)
        obs, r, done, _ = problem.step(u[t])

        if done:
            break

            # alternatively, continue
            print("done! problem resetting...")
            obs = problem.initialize()
            u = model.ilqr(obs, H, threshold, lamb, max_iterations)

    problem.close()
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
