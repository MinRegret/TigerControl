import tigercontrol
from tigercontrol.models.optimizers.ons import ONS
from tigercontrol.models.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_ons(show=False):

    problem = tigercontrol.problem('ARMA-v0')
    x = problem.initialize(p=2,q=0)

    model = tigercontrol.model('LSTM')
    model.initialize(n=1, m=1, l=5, h=10, optimizer=ONS) # initialize with class
    model.predict(1.0) # call methods to verify it works
    model.update(1.0)

    optimizer = ONS(learning_rate=0.001)
    model = tigercontrol.model('LSTM')
    model.initialize(n=1, m=1, l=3, h=10, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = model.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        model.update(y_true)
        x = y_true

    if show:
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_ons passed")



if __name__ == "__main__":
    test_ons(show=True)