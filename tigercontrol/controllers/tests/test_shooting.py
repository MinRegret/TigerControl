# Test the ODEShootingController controller class by solving x'' = x + 4exp(t) for L = 1

import tigercontrol
import jax.numpy as np
import math
import matplotlib.pyplot as plt

def test_shooting(steps=1000, show=False):
    env = tigercontrol.environment("CartPole-v0")
    obs = env.initialize()
    n, m, T = 4, 1, 10 # dimensions of obs, actions

    shooting = tigercontrol.controller("Shooting")
    shooting.initialize(n, m, T, env, optimizer=None, update_steps=25, learning_rate=0.01)

    count = 0
    for t in range(steps):
        count += 1
        u = shooting.get_action(obs)
        obs, r, done = env.step(u)
        if show: 
            env.render()
            time.sleep(1. / 50.)
        if done:
            if show:
                print("lasted {} time steps".format(count))
                count = 0
            obs = env.initialize()
    env.close()
    print("test_shooting passed")
    return

if __name__=="__main__":
    test_shooting(show=True)