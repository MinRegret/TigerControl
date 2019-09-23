"""An example to run of the minitaur gym environment with sine gaits.

"""

import tigercontrol
import numpy as np
import time

def test_minitaur(steps=1000, verbose=False):
    problem = tigercontrol.problem("PyBullet-Minitaur-v0")
    problem.initialize(render=verbose)

    sum_reward = 0
    amplitude1 = 0.5
    amplitude2 = 0.5
    speed = 40

    for step_counter in range(steps):
        time_step = 0.01
        t = step_counter * time_step
        a1 = np.sin(t * speed) * amplitude1
        a2 = np.sin(t * speed + np.pi) * amplitude1
        a3 = np.sin(t * speed) * amplitude2
        a4 = np.sin(t * speed + np.pi) * amplitude2
        action = [a1, a2, a2, a1, a3, a4, a4, a3]
        _, reward, done, _ = problem.step(action)
        time.sleep(1. / 100.)

        sum_reward += reward
        if done:
            problem.reset()

    problem.close()
    print("test_minitaur passed")


if __name__ == '__main__':
    test_minitaur(verbose=True)


