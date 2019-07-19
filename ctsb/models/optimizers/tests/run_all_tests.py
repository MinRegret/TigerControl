from ctsb.models.optimizers.tests.test_adagrad import test_adagrad
from ctsb.models.optimizers.tests.test_sgd import test_sgd


# run all optimizers tests
def run_all_tests(steps=1000, show=False):
    print("\nrunning all optimizers tests...\n")
    test_sgd(time=1)
    test_adagrad(time=1)
    print("\nall optimizers tests passed\n")


if __name__ == "__main__":
    run_all_tests(show=True)
