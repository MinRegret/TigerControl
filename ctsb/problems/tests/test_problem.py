# test the Random problem class

from ctsb import core
import jax.numpy as np
import matplotlib.pyplot as plt


class TestClass(core.Problem):
    def __init__(self):
        self.T = 0

    def step(self):
        self.T += 1
        return np.random.normal(size=(1,))

    def reset(self):
        self.T = 0


def test_random():
    random = ctsb.Random()
    assert random.T == 0

    test_output = []
    for t in range(1000):
        test_output.append(random.step())

    assert random.T == 1000
    plt.plot(test_output)
    plt.show()
    plt.pause(5)
    plt.close()
    return


if __name__=="__main__":
    test_random()