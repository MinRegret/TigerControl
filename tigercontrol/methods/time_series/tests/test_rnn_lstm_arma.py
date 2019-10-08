# test the RNN and LSTM method classes on ARMA

import tigercontrol
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigercontrol.utils import generate_key

def test_rnn_lstm_arma(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 0
    problem = tigercontrol.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)

    method_RNN = tigercontrol.method("RNN")
    method_RNN.initialize(1, 1, l = p)

    method_LSTM = tigercontrol.method("LSTM")
    method_LSTM.initialize(1, 1, l = p)

    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results_RNN = []
    results_LSTM = []
    for i in range(T):
        y_pred_RNN = method_RNN.predict(cur_x)
        y_pred_LSTM = method_LSTM.predict(cur_x)
        if(i == 0):
            print(method_RNN.forecast(cur_x, timeline = 10))
            print(method_LSTM.forecast(cur_x, timeline = 10))
        y_true = problem.step()
        results_RNN.append(loss(y_true, y_pred_RNN))
        results_LSTM.append(loss(y_true, y_pred_LSTM))
        cur_x = y_true
        method_RNN.update(y_true)
        method_LSTM.update(y_true)

    if show_plot:
        plt.plot(results_RNN, label = 'RNN')
        plt.plot(results_LSTM, label = 'LSTM')
        plt.legend()
        plt.title("RNN vs. LSTM on ARMA problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_rnn_lstm_arma passed")
    return

if __name__=="__main__":
    test_rnn_lstm_arma()