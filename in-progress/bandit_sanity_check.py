""" let's see if bandit optimization works! """

import sys
import os
import time

import jax 
import jax.numpy as np
import jax.random as random
from tigercontrol.utils import generate_key, set_key

import matplotlib.pyplot as plt

# bandit memory implementation
from controllers.bandit_memory import BanditMemory

if __name__ == "__main__":

    # fix key for reproducibility
    set_key(0)

    # hyperparameters
    T = 10000
    d, H = 3, 3
    delta = 0.05
    initial_lr = 0.002
    mag = 10
    x_init = mag * np.ones((d, H))
    loss = lambda x: np.sum(np.sum(x, axis=0)**2) / (mag*H*np.sqrt(d)) # loss(x_t-H+1, ..., x_t) = C*||sum_i x_t-i||^2 

    print("\nStarting bandit sanity check with T = {}!".format(T))
    print("x_H initialized with norm {}\n".format(np.linalg.norm(x_init[-1])))
    bandit = BanditMemory(x_init=x_init, d=d, H=H, f=loss, delta=delta, initial_lr=initial_lr, magnitude=(mag*np.sqrt(d)))

    print_num_times = 50

    losses = []
    xs = []
    ys = []
    for t in range(T):
        x_t, y_t, loss_t = bandit.step() # use same f=loss at every time step
        #xs.append(x_t)
        #ys.append(y_t)
        losses.append(loss_t)
        if T > 100 and ((print_num_times * (t+1)) % (print_num_times * int(T/print_num_times)) == 0):
            print("{}% done, t = {}, x_t norm = {}".format(100*t/T, t, np.linalg.norm(x_t)))

    plt.plot(losses)
    plt.title("Bandit with Memory sanity check")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

