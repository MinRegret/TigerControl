import tigercontrol
from tigercontrol.utils.optimizers.sgd import SGD
from tigercontrol.utils.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_sgd(show=False):
    test_sgd_lstm(show=show)
    test_sgd_autoregressor(show=show)
    print("test_sgd passed")

def test_sgd_lstm(show=False):
    environment = tigercontrol.environment('ARMA')
    x = environment.initialize(p=2,q=0)

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=3, h=10, optimizer=SGD) # initialize with class
    controller.predict(1.0) # call controllers to verify it works
    controller.update(1.0)

    optimizer = SGD(learning_rate=0.001)
    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=3, h=10, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = controller.predict(x)
        y_true = environment.step()
        loss.append(mse(y_pred, y_true))
        controller.update(y_true)
        x = y_true

    if show:
        plt.title("Test SGD on ARMA(3) with LSTM controller")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def test_sgd_autoregressor(show=False):
    environment = tigercontrol.environment('ARMA')
    x = environment.initialize(p=2,q=0)

    optimizer = SGD(learning_rate=0.0003)
    controller = tigercontrol.controllers('AutoRegressor')
    controller.initialize(p=3, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = controller.predict(x)
        y_true = environment.step()
        loss.append(mse(y_pred, y_true))
        controller.update(y_true)
        x = y_true

    if show:
        plt.title("Test SGD on ARMA(3) with AutoRegressor controller")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()



if __name__ == "__main__":
    test_sgd(show=True)