import tigercontrol
from tigercontrol.utils.optimizers.ons import ONS
from tigercontrol.utils.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_ons(show=False):

    environment = tigercontrol.environment('ARMA')
    x = environment.reset(p=2,q=0)

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=5, h=10, optimizer=ONS) # initialize with class
    controller.predict(1.0) # call controllers to verify it works
    controller.update(1.0)

    optimizer = ONS(learning_rate=0.001)
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
    print("test_ons passed")



if __name__ == "__main__":
    test_ons(show=True)