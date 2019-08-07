import ctsb
from ctsb.models.optimizers.ons import ONS
from ctsb.models.optimizers.deprecated_ons import deprecated_ONS
from ctsb.models.optimizers.losses import mse
import matplotlib.pyplot as plt
import time

def test_ons(show=False):
    problem = ctsb.problem('ARMA-v0')
    x = problem.initialize(p=2,q=0)

    model1 = ctsb.model('LSTM')
    model1.initialize(n=1, m=1, l=3, h=10, optimizer=ONS) # initialize with class
    model1 = ctsb.model('AutoRegressor')
    model1.initialize(optimizer=ONS) # initialize with class

    model1.predict(1.0) # call methods to verify it works
    model1.update(1.0)

    model2 = ctsb.model('LSTM')
    model2.initialize(n=1, m=1, l=3, h=10, optimizer=deprecated_ONS) # initialize with class

    model2.predict(1.0) # call methods to verify it works
    model2.update(1.0)

    loss1, loss2 = [], []
    time_start = time.time()
    for t in range(100):
        y_pred1 = model1.predict(x)
        y_pred2 = model2.predict(x)
        y_true = problem.step()
        loss1.append(mse(y_pred1, y_true))
        model1.update(y_true)
        loss2.append(mse(y_pred2, y_true))
        model2.update(y_true)
        x = y_true
    print(time.time() - time_start)

    if show:
        plt.plot(loss1, label = 'ONS')
        plt.plot(loss2, label = 'deprecated ONS')
        plt.legend()
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
    print("test_ons passed")

if __name__ == "__main__":
    test_ons(show=True)