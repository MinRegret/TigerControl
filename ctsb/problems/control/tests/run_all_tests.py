from ctsb.problems.control.tests.test_lds import test_lds
from ctsb.problems.control.tests.test_lstm_output import test_lstm
from ctsb.problems.control.tests.test_rnn_output import test_rnn
from ctsb.problems.control.tests.test_cartpole import test_cartpole
from ctsb.problems.control.tests.test_cartpole_double import test_cartpole_double
from ctsb.problems.control.tests.test_cartpole_swingup import test_cartpole_swingup
from ctsb.problems.control.tests.test_simulator_wrapper import test_simulator_wrapper

def run_all_tests(steps=1000, show=False):
    print("\nrunning all control problems tests...\n")
    test_lds(steps=steps, show_plot=show)
    test_lstm(steps=steps, show_plot=show)
    test_rnn(steps=steps, show_plot=show)
    print("\nall control problems tests passed\n")
    print("\nrunning all pybullet problems tests...\n")
    test_cartpole(verbose=show)
    test_cartpole_swingup(verbose=show)
    test_cartpole_double(verbose=show)
    test_simulator_wrapper(verbose=show)
    print("\nall pybullet problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=True)