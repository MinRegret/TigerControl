"""
An example to run of the half cheetah gym environment with random gaits.
"""

import tigercontrol
import numpy as np
import time

def test_ant(steps=1000, verbose=False):
    problem = tigercontrol.problem("PyBullet-Ant-v0")
    problem.initialize(render=verbose)

    sum_reward = 0
    amplitude1 = 0.5
    amplitude2 = 0.5
    speed = 40

    action = np.random.normal(size=problem.action_space)

    for step_counter in range(steps):
        action = 0.95 * action + np.random.normal(size=problem.action_space)
        _, reward, done, _ = problem.step(action)
        time.sleep(1. / 100.)

        sum_reward += reward
        if done:
            problem.reset()
    print("test_ant passed")


if __name__ == '__main__':
    test_ant(verbose=True)


