# test the LSTM model class

import ctsb
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from ctsb.utils import generate_key

def test_lstm(steps=100, show_plot=True):
    T = steps 
    n, m, l, d = 4, 5, 10, 10
    problem = ctsb.problem("LDS-Control-v0")
    y_true = problem.initialize(n, m, d)
    model = ctsb.model("LSTM")
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
        plt.title("LSTM model on LDS problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_lstm passed")
    return

if __name__=="__main__":
    test_lstm()