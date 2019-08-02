"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np

# cartpole test
def test_cartpole(verbose=False):
    problem = ctsb.problem("CartPole-v0")
    obs, r, done, _ = problem.initialize()

    t_start = time.time()
    save_to_mem_ID = -1

    frame = 0
    score = 0
    restart_delay = 0
    saved = False
    while time.time() - t_start < 3:
        if verbose:
            problem.render()
        time.sleep(1. / 60.)
        a = 0.0 # no action
        obs, r, done, _ = problem.step(a)

        score += r
        frame += 1
        if done:
            obs, r, done, _ = problem.initialize()
            
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
