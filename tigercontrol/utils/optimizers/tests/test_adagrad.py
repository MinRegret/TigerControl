import tigercontrol
from tigercontrol.utils.optimizers.adagrad import Adagrad
from tigercontrol.utils.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_adagrad(show=False):

    environment = tigercontrol.environment('ARMA-v0')
    x = environment.initialize(p=2,q=0)

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=3, h=10, optimizer=Adagrad) # initialize with class
    controller.predict(1.0) # call controllers to verify it works
    controller.update(1.0)

    optimizer = Adagrad(learning_rate=0.1)
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
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_adagrad passed")



if __name__ == "__main__":
    test_adagrad(show=True)