# test the RNN Output problem class

import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import ctsb
from ctsb.problems.simulated.rnn_output import RNN_Output
from ctsb.utils.random import generate_key



def test_rnn(steps=100, show_plot=False, verbose=False):
    T = steps
    n, m, l, h = 5, 3, 5, 10
    problem = RNN_Output()
    problem.initialize(n, m, l, h)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        if verbose and (t+1) * 10 % T == 0:
            print("{} timesteps".format(t+1))
        u = random.normal(generate_key(),shape=(n,))
        test_output.append(problem.step(u))

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("rnn")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_rnn passed")
    return


if __name__=="__main__":
    test_rnn()