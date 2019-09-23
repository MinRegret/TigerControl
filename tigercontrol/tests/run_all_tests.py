"""
Run all tests for the TigerControl framework
"""

from tigercontrol.utils.tests.run_all_tests import run_all_tests as utils_tests
from tigercontrol.problems.tests.run_all_tests import run_all_tests as problems_tests
from tigercontrol.models.tests.run_all_tests import run_all_tests as models_tests
from tigercontrol.experiments.tests.run_all_tests import run_all_tests as experiments_tests
from tigercontrol.tests.test_tigercontrol_functionality import test_tigercontrol_functionality

# run all sub-level tests
def run_all_tests(show_results=False):

    print("\n----- Running all TigerControl tests! -----\n")

    utils_tests(show=show_results)
    experiments_tests(show=show_results)
    models_tests(show=show_results)
    problems_tests(show=show_results)
    test_tigercontrol_functionality()

    print("\n----- Tests done -----\n")

if __name__ == "__main__":
    run_all_tests(show_results=False)