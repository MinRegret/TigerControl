# test the Random problem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt


def test_random(steps=1000, show_plot=False):
    T = steps
    problem = tigercontrol.problem("Random-v0")
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("random")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_random passed")
    return


if __name__=="__main__":
    test_random()