"""
Autoregressive moving-average
"""

import jax
import jax.numpy as np
import jax.random as random
from jax import jit
from tqdm import tqdm
import matplotlib.pyplot as plt

class ARMA():
    """
    Simulates an autoregressive moving-average time-series.
    """

    def __init__(self):
        self.key = random.PRNGKey(0)

    def generate_key(self):
        key, subkey = random.split(self.key)
        self.key = key
        return subkey

    def initialize(self, p, q, c=None):
        """
        Description:
            Randomly initialize the hidden dynamics of the system.
        Args:
            p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
                initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
                If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
            q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
                initializes a Gaussian length-q vector (no bound on norm). If p is a
                1-dimensional numpy.ndarray then uses it as dynamics vector.
            c (float): Default value follows a normal distribution. The ARMA dynamics 
                follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
                to be centered around mean c.
        Returns:
            The first value in the time-series
        """
        self.T = 0
        phi = random.normal(self.generate_key(), shape=(p,))
        self.phi = 1.0 * phi / np.linalg.norm(phi, ord=1)
        self.psi = random.normal(self.generate_key(), shape=(q,))
        self.p = self.phi.shape[0]
        self.q = self.psi.shape[0]
        self.c = random.normal(self.generate_key()) if c == None else c
        self.x = random.normal(self.generate_key(), shape=(self.p,))
        self.noise = random.normal(self.generate_key(), shape=(q,))

        self.f = jit(self._x_noise_update)
        return self.x[0]

    def _x_noise_update(self, self_x, self_noise, key):
        x_ar = np.dot(self_x, self.phi)
        x_ma = np.dot(self_noise, self.psi)
        eps = random.normal(key)
        x_new = self.c + x_ar + x_ma + eps

        next_x = jax.ops.index_update(self_x, jax.ops.index[1:], self_x[:-1])
        next_noise = jax.ops.index_update(self_noise, jax.ops.index[1:], self_noise[:-1])

        next_x = jax.ops.index_update(self_x, 0, x_new) # equivalent to self.x[0] = x_new
        next_noise = jax.ops.index_update(self_noise, 0, eps) # equivalent to self.noise[0] = eps
        return (next_x, next_noise, x_new)

    def step(self):
        """
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the ARMA time-series.
        """
        self.T += 1
        self.x, self.noise, x_new = self.f(self.x, self.noise, self.generate_key())   
        return x_new


def main():
    T = 1000
    p, q = 3, 3
    problem = ARMA()
    problem.initialize(p,q)

    test_output = []
    print("ARMA test start for T = {}".format(T))
    for t in tqdm(range(T)):
        test_output.append(problem.step())
    print("ARMA test complete")
    plt.plot(test_output)
    plt.title("arma")
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    return


if __name__ == "__main__":
    main()


