# test the MPPI model class

import ctsb
import jax.numpy as np
import jax.random as random
from ctsb.utils import generate_key
import matplotlib.pyplot as plt

def test_mppi(steps=10, show_plot=True):

    T = 20 
    K = steps

    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 1

    U = [i for i in random.uniform(generate_key(), minval = ACTION_LOW, maxval = ACTION_HIGH, shape = (T,))]  # pendulum joint effort in (-2, +2)

    problem = ctsb.problem("CartPole-v0")
    problem.initialize()

    model = ctsb.model("MPPI")
    model.initialize(env = problem, K = K, T = T, U = U, lambda_ = lambda_, u_init = 0)
    model.step(n = 10)
 
    print("test_mppi passed")
    return

if __name__=="__main__":
    test_mppi()