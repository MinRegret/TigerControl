from tigercontrol.problems.control.tests.run_all_tests import run_all_tests as control_problems_tests
from tigercontrol.problems.time_series.tests.run_all_tests import run_all_tests as time_series_problems_tests
from tigercontrol.problems.pybullet.tests.run_all_tests import run_all_tests as pybullet_problems_tests
from tigercontrol.problems.tests.test_custom_problem import test_custom_problem

# run all unit tests for problems
def run_all_tests(steps=1000, show=False):
    print("\nrunning all problems tests...\n")
    control_problems_tests(show=show)
    time_series_problems_tests(show=show)
    pybullet_problems_tests(show=show)
    test_custom_problem(show=show)
    print("\nall problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)