""" boosting tests """
from tigercontrol.controllers.boosting.tests.test_simple_boost import test_simple_boost

# run all boosting tests
def run_all_tests(steps=1000, show=False):
    print("\nrunning all boosting tests...\n")
    test_simple_boost(steps=steps, show=show)
    print("\nall boosting tests passed\n")


if __name__ == "__main__":
    run_all_tests(show=False)
