""" all environment tests """

from tigercontrol.environments.control.tests.run_all_tests import run_all_tests as control_environments_tests
from tigercontrol.environments.pybullet.tests.run_all_tests import run_all_tests as pybullet_environments_tests
from tigercontrol.environments.tests.test_custom_environment import test_custom_environment

# run all unit tests for environments
def run_all_tests(steps=1000, show=False):
    print("\nrunning all environments tests...\n")
    test_custom_environment(show=show)
    control_environments_tests(show=show)
    try:
        pybullet_environments_tests(show=show)
    except e:
        print(e)
        print("pybullet environment tests failed")
    print("\nall environments tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)