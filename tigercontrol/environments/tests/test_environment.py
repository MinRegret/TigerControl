"""
Test for all TigerControl environments
"""
import time
import tigercontrol
import jax.numpy as np
import sys
import argparse


def test_environments(stress=True):
    """ Description: test all environments """
    env_ids = tigercontrol.environment_registry.list_ids() # get all controllers

    for env_id in env_ids:
        print("\n------ {} tests -----\n".format(env_id))
        try:
            test_environment(env_id, stress=stress)
        except Exception as e:
            print("environment {} raised error {}".format(env_id, e))


def test_environment(env_id, stress=True):
    """ Description: run test for a single environment """
    test_api(env_id)
    if stress: 
        test_stress(env_id)


def test_api(env_id):
    """ Description: verify that all default methods work for this environment """
    environment_class = tigercontrol.environment(env_id)
    env = environment_class()
    n, m = env.get_state_dim(), env.get_action_dim() # get dimensions of system
    x_0 = env.reset() # first state
    x_1 = env.step(np.zeros(m)) # try step with 0 action
    assert np.isclose(x_0, env.reset()) # assert reset return back to original state


def test_stress(env_id):
    """ Description: run environment in a few high pressure situations """
    environment_class = tigercontrol.environment(env_id)
    env = environment_class()
    n, m = env.get_state_dim(), env.get_action_dim() # get dimensions of system
    x_0 = env.reset()
    for t in range(10000):
        env.step(t * np.ones(m))



# main file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", type=bool, help="perform stress test", default=True)
    parser.add_argument("--env", type=str, help="environment id/name", default="all")
    args = parser.parse_args()


    # if no env specified, run tests for all envs
    if args.env == "all":
        test_environments(stress=args.stress)
    # check specified env is in registry and run individual tests
    else:
        if args.env not in tigercontrol.environment_registry.list_ids():
            raise Exception("{} is not a valid env id".format(args.env))
        test_environment(args.env, args.stress)


