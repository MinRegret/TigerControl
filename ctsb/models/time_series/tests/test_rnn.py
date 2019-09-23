# test the RNN model class

import tigercontrol
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigercontrol.utils import generate_key

def test_rnn(steps=100, show_plot=True):
    T = steps 
    n, m, l, d = 4, 5, 10, 10
    problem = tigercontrol.problem("LDS-Control-v0")
    y_true = problem.initialize(n, m, d)
    model = tigercontrol.model("RNN")
    model.initialize(n, m, l, d)
    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results = []
    for i in range(T):
        u = random.normal(generate_key(), (n,))
        y_pred = model.predict(u)
        y_true = problem.step(u)
        results.append(loss(y_true, y_pred))
        model.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("RNN model on LDS problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_rnn passed")
    return

if __name__=="__main__":
    test_rnn()