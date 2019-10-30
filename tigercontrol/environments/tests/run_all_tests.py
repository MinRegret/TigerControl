from tigercontrol.environments.controller.tests.run_all_tests import run_all_tests as control_environments_tests
from tigercontrol.environments.time_series.tests.run_all_tests import run_all_tests as time_series_environments_tests
from tigercontrol.environments.pybullet.tests.run_all_tests import run_all_tests as pybullet_environments_tests
from tigercontrol.environments.tests.test_custom_environment import test_custom_environment

# run all unit tests for environments
def run_all_tests(steps=1000, show=False):
    print("\nrunning all environments tests...\n")
    control_environments_tests(show=show)
    time_series_environments_tests(show=show)
    pybullet_environments_tests(show=show)
    test_custom_environment(show=show)
    print("\nall environments tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)