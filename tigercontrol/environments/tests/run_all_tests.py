from tigercontrol.environments.tests.test_lds import test_lds
from tigercontrol.environments.tests.test_lstm_control import test_lstm_control
from tigercontrol.environments.tests.test_rnn_control import test_rnn_control
from tigercontrol.environments.tests.test_cartpole import test_cartpole
from tigercontrol.environments.tests.test_pendulum import test_pendulum
from tigercontrol.environments.tests.test_double_pendulum import test_double_pendulum
from tigercontrol.environments.tests.test_obstacles_improved import test_obstacles_improved
from tigercontrol.environments.tests.test_planar_quadrotor import test_planar_quadrotor
from tigercontrol.environments.tests.test_double_pendulum import test_double_pendulum


def run_all_tests(steps=1000, show=False):
    print("\nrunning all control environments tests...\n")
    test_lstm_control(steps=steps, show_plot=show)
    test_rnn_control(steps=steps, show_plot=show)
    test_cartpole(verbose=show)
    test_pendulum(verbose=show)
    test_double_pendulum(verbose=show)
    test_double_pendulum(verbose=show)
    test_obstacles_improved(verbose=show)
    test_planar_quadrotor(steps=steps, show_plot=show)
    test_lds(steps=steps, show_plot=show)
    print("\nall control environments tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)