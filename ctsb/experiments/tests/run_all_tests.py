from ctsb.experiments.tests.test_experiment import test_experiment

# add all unit tests in dataset_registry
def run_all_tests(show=False):
    print("\nrunning all experiments tests...\n")
    test_experiment(show=show)
    print("\nall experiments tests passed\n")

if __name__ == '__main__':
    run_all_tests(show=True)