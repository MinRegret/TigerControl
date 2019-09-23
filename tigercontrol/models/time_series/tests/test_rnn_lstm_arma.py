# test the RNN and LSTM model classes on ARMA

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

    model_RNN = tigercontrol.model("RNN")
    model_RNN.initialize(1, 1, l = p)

    model_LSTM = tigercontrol.model("LSTM")
    model_LSTM.initialize(1, 1, l = p)

    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results_RNN = []
    results_LSTM = []
    for i in range(T):
        y_pred_RNN = model_RNN.predict(cur_x)
        y_pred_LSTM = model_LSTM.predict(cur_x)
        if(i == 0):
            print(model_RNN.forecast(cur_x, timeline = 10))
            print(model_LSTM.forecast(cur_x, timeline = 10))
        y_true = problem.step()
        results_RNN.append(loss(y_true, y_pred_RNN))
        results_LSTM.append(loss(y_true, y_pred_LSTM))
        cur_x = y_true
        model_RNN.update(y_true)
        model_LSTM.update(y_true)

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