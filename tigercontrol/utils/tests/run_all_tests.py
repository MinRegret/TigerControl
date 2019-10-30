import os
import re
from tigercontrol.utils.tests.test_random import test_random
from tigercontrol.utils.tests.test_registration_tools import test_registration_tools
from tigercontrol.utils.optimizers.tests.run_all_tests import run_all_tests as optimizers_tests
from tigercontrol.utils.boosting.tests.run_all_tests import run_all_tests as boosting_tests

# add all unit tests in utils
def run_all_tests(steps=1000, show=False):
    print("\nrunning all utils tests...\n")
    test_random(show=show)
    test_registration_tools(show=show)
    optimizers_tests(steps=steps, show=show)
    boosting_tests()
    print("\nall utils tests passed\n")

if __name__ == '__main__':
    run_all_tests(show=False)