""" all autotuning tests """
from tigercontrol.utils.autotuning.tests.test_grid_search import test_grid_search

# run all autotuning tests
def run_all_tests(show=False):
    print("\nrunning all autotuning tests...\n")
    test_grid_search(show=show)
    print("\nall autotuning tests passed\n")


if __name__ == "__main__":
    run_all_tests(show=True)
