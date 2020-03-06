"""
Test for all TigerControl controllers
"""
import time
import tigercontrol
import jax.numpy as np
import sys
import argparse


def test_controllers(stress=True):
    """ Description: test all controllers """
    ctrl_ids = tigercontrol.controller_registry.list_ids() # get all controllers

    for ctrl_id in ctrl_ids:
        print("\n------ {} tests -----\n".format(ctrl_id))
        test_controller(ctrl_id, stress=stress)


def test_controller(ctrl_id, stress=True):
    """ Description: run test for a single controller """
    try:
        test_api(ctrl_id)
        if stress: test_stress(ctrl_id)
    except Exception as e:
        print("controller {} raised error {}".format(ctrl_id, e))


def test_api(ctrl_id):
    """ Description: verify that all default methods work for this controller """
    controller_class = tigercontrol.controller(ctrl_id)
    n, m = 3, 2 # arbitrary state/action dimensions
    ctrl = controller_class()
    n, m = ctrl.get_state_dim(), ctrl.get_action_dim() # get dimensions of system
    x_0 = ctrl.reset() # first state
    x_1 = ctrl.step(np.zeros(m)) # try step with 0 action
    assert np.isclose(x_0, ctrl.reset()) # assert reset return back to original state


def test_stress(ctrl_id):
    """ Description: run controller in a few high pressure situations """
    controller_class = tigercontrol.controller(ctrl_id)
    ctrl = controller_class()
    n, m = ctrl.get_state_dim(), ctrl.get_action_dim() # get dimensions of system
    x_0 = ctrl.reset()
    for t in range(10000):
        ctrl.step(t * np.ones(m))



# main file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", type=bool, help="perform stress test", default=True)
    parser.add_argument("--ctrl", type=str, help="controller id/name", default="all")
    args = parser.parse_args()


    # if no ctrl specified, run tests for all ctrls
    if args.ctrl == "all":
        test_controllers(stress=args.stress)
    # check specified ctrl is in registry and run individual tests
    else:
        if args.ctrl not in tigercontrol.controller_registry.list_ids():
            raise Exception("{} is not a valid ctrl id".format(args.ctrl))
        test_controller(args.ctrl, args.stress)


