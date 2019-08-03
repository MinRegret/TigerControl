"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np


# cartpole test
def test_cartpole(verbose=False):
    problem = ctsb.problem("CartPole-v0")
    obs = problem.initialize()
    print(obs)

    model = ctsb.model("ILQR")
    loss = lambda obs, a : obs[2]**2
    dyn = problem.dynamics
    x_0 = obs
    T = 100
    lamb = 0.1

    u = model.ilqr(1, dyn, loss, x_0, T, 0.05, lamb)

    # t_start = time.time()
    # save_to_mem_ID = -1

    frame = 0
    score = 0
    restart_delay = 0
    saved = False
    # while time.time() - t_start < 3:
    for t in range(T):
        if verbose:
            problem.render()
        time.sleep(1. / 6.)
        a = u[t] # no action
        obs, r, done, _ = problem.step(a)
        # print(obs)
        print(a)

        score += r
        frame += 1
        '''
        if done:
            obs, r, done, _ = problem.initialize()
        '''
            
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)
