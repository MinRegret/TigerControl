"""
Run all tests for the CTSB framework
"""

from ctsb.utils.tests import run_all_tests as utils_tests

# run all sub-level tests
def run_all_tests():
    utils_tests()


if __name__ == "__main__":
    test_utils()