# run all experiments tests

from tigercontrol.experiments.tests.test_new_experiment import test_new_experiment
from tigercontrol.experiments.tests.test_precomputed import test_precomputed

# add all unit tests in dataset_registry
def run_all_tests(show=False):
    print("\nrunning all experiments tests...\n")
    test_new_experiment(show=show)
    test_precomputed(show=show)
    print("\nall experiments tests passed\n")

if __name__ == '__main__':
    run_all_tests(show=True)