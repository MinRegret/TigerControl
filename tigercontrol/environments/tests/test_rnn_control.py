# test the RNN Output environment class

import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import tigercontrol
from tigercontrol.utils.random import generate_key



def test_rnn_control(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m = 5, 3
    environment = tigercontrol.environment("RNN-Control-v0")
    environment.initialize(n, m)
    assert environment.T == 0

    test_output = []
    for t in range(T):
        if verbose and (t+1) * 10 % T == 0:
            print("{} timesteps".format(t+1))
        u = random.normal(generate_key(),shape=(n,))
        test_output.append(environment.step(u))

    info = environment.hidden()
    if verbose:
        print(info)

    assert environment.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("rnn")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_rnn_control passed")
    return


if __name__=="__main__":
    test_rnn_control(show_plot=True)