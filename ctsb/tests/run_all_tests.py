"""
Run all tests for the CTSB framework
"""

from ctsb.utils.tests.run_all_tests import run_all_tests as utils_tests
from ctsb.problems.tests.run_all_tests import run_all_tests as problems_tests
from ctsb.models.tests.run_all_tests import run_all_tests as models_tests

# run all sub-level tests
def run_all_tests():
    utils_tests()
    problems_tests()
    models_tests()


if __name__ == "__main__":
    run_all_tests()