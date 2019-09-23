# test the UCI Indoor problem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt


def test_uci_indoor(steps=1000, show_plot=False, verbose=False):
    T = steps
    problem = tigercontrol.problem("UCI-Indoor-v0")
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step()[1])

    assert problem.T == T
    if verbose:
        print(problem.hidden())
    if show_plot:
        plt.plot(test_output)
        plt.title("UCI Indoor")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    print("test_uci_indoor passed")
    return


if __name__=="__main__":
    test_uci_indoor(show_plot=True, verbose=True)