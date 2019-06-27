# test the PredictZero model class

import ctsb
import jax.numpy as np
import matplotlib.pyplot as plt

def test_last_value(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 3
    problem = ctsb.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)
    model = ctsb.model("LastValue")
    model.initialize()
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
 
    results = []
    for i in range(T):
        cur_y_pred = model.predict(cur_x)
        cur_y_true = problem.step()
        cur_loss = loss(cur_y_true, cur_y_pred)
        results.append(cur_loss)
        model.update(cur_loss)
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results)
        plt.title("LastValue model on ARMA problem")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_last_value passed")
    return

if __name__=="__main__":
    test_last_value()