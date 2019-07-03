# test the Autogressor model class

import ctsb
import jax.numpy as np
import matplotlib.pyplot as plt
ctsb.set_key(0)

def test_autoregressor(steps=100, show_plot=True):
    T = steps 
    p = 3
    problem = ctsb.problem("SP500-v0")
    cur_x = problem.initialize() / 300
    model_AR = ctsb.model("AutoRegressor")
    model_AR.initialize(p)
    model_LV = ctsb.model("LastValue")
    model_LV.initialize()
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
 
    results_AR = []
    results_LV = []
    for i in range(T):
        cur_y_pred_AR = model_AR.predict(cur_x)
        cur_y_pred_LV = model_LV.predict(cur_x)
        cur_y_true = problem.step() / 300
        cur_loss_AR = loss(cur_y_true, cur_y_pred_AR)
        cur_loss_LV = loss(cur_y_true, cur_y_pred_LV)
        results_AR.append(cur_loss_AR)
        results_LV.append(cur_loss_LV)
        model_AR.update(cur_loss_AR)
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results_AR)
        #plt.plot(results_LV)
        plt.title("Autoregressive vs. LastValue model on SP500 problem")
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    print("test_autoregressor passed")
    return

if __name__=="__main__":
    test_autoregressor()