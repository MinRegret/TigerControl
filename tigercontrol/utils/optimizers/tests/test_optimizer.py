"""
Test for all TigerControl optimizers
"""
import time
import tigercontrol
import jax.numpy as np
import sys
import argparse

from tigercontrol.utils.optimizers.sgd import SGD
from tigercontrol.utils.optimizers.ogd import OGD
from tigercontrol.utils.optimizers.ons import ONS
from tigercontrol.utils.optimizers.adam import Adam
from tigercontrol.utils.optimizers.adagrad import Adagrad

""" list all optimizers """
optimizers = {
    'Adagrad': Adagrad,
    'Adam': Adam,
    'OGD': OGD,
    'ONS': ONS,
    'SGD': SGD,
}


def test_optimizers(stress=True):
    """ Description: test all optimizers """
    opt_ids = tigercontrol.optimizer_registry.list_ids() # get all controllers

    for opt_id in opt_ids:
        print("\n------ {} tests -----\n".format(opt_id))
        test_optimizer(opt_id, stress=stress)


def test_optimizer(opt_id, stress=True):
    """ Description: run test for a single optimizer """
    try:
        test_api(opt_id)
        if stress: test_stress(opt_id)
    except Exception as e:
        print("optimizer {} raised error {}".format(opt_id, e))


def test_api(opt_id):
    """ Description: verify that all default methods work for this optimizer """
    optimizer_class = tigercontrol.optimizer(opt_id)
    opt = optimizer_class()
    n, m = opt.get_state_dim(), opt.get_action_dim() # get dimensions of system
    x_0 = opt.reset() # first state
    x_1 = opt.step(np.zeros(m)) # try step with 0 action
    assert np.isclose(x_0, opt.reset()) # assert reset return back to original state


def test_stress(opt_id):
    optironment = tigercontrol.optironment('ARMA')
    x = optironment.reset(p=2,q=0)

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=5, h=10, optimizer=OGD) # initialize with class
    controller.predict(1.0) # call controllers to verify it works
    controller.update(1.0)

    optimizer = OGD(learning_rate=0.001)
    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n=1, m=1, l=3, h=10, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = controller.predict(x)
        y_true = optironment.step()
        loss.append(mse(y_pred, y_true))
        controller.update(y_true)
        x = y_true

    if show:
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_ogd passed")


# main file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", type=bool, help="perform stress test", default=True)
    parser.add_argument("--opt", type=str, help="optimizer id/name", default="all")
    args = parser.parse_args()


    # if no opt specified, run tests for all opts
    if args.opt == "all":
        test_optimizers(stress=args.stress)
    # check specified opt is in registry and run individual tests
    else:
        if args.opt not in tigercontrol.optimizer_registry.list_ids():
            raise Exception("{} is not a valid opt id".format(args.opt))
        test_optimizer(args.opt, args.stress)


