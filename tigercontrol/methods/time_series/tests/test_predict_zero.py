# test the PredictZero method class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt

def test_predict_zero(steps=1000, show_plot=True):
    T = steps 
    p, q = 3, 3
    problem = tigercontrol.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)
    method = tigercontrol.method("PredictZero")
    method.initialize()
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
 
    results = []
    for i in range(T):
        cur_y_pred = method.predict(cur_x)
        #print(method.forecast(cur_x, 10))
        cur_y_true = problem.step()
        cur_loss = loss(cur_y_true, cur_y_pred)
        results.append(cur_loss)
        method.update(cur_y_true)
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results)
        plt.title("Zero method on ARMA problem")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_predict_zero passed")
    return

if __name__=="__main__":
    test_predict_zero()