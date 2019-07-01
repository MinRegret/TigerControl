"""
Run all tests for the CTSB framework
"""

from ctsb.utils.tests.run_all_tests import run_all_tests as utils_tests
from ctsb.problems.tests.run_all_tests import run_all_tests as problems_tests
from ctsb.models.tests.run_all_tests import run_all_tests as models_tests

# run all sub-level tests
def run_all_tests():

    print("\n----- Running all CTSB tests! -----\n")

    utils_tests()
    problems_tests()
    models_tests()

    print("\n----- Tests done -----\n")

if __name__ == "__main__":
    run_all_tests()