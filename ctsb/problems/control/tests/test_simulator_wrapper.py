"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np
from ctsb.models.control.control_model import ControlModel
from cartpole_weights import *
from ctsb.problems.control.pybullet.simulator_wrapper import SimulatorWrapper


# neural network policy class trained specifically for the cartpole problem
class SmallReactivePolicy(ControlModel):
    "Simple multi-layer perceptron policy, no internal state"

    def __init__(self):
        self.initialized = False

    def initialize(self, observation_space, action_space):
        self.initialized = True
        assert weights_dense1_w.shape == (observation_space.shape[0], 64.0)
        assert weights_dense2_w.shape == (64.0, 32.0)
        assert weights_final_w.shape == (32.0, action_space.shape[0])

    def predict(self, ob): # weights can be fount at the end of the file
        x = ob
        x = np.maximum((np.dot(x, weights_dense1_w) + weights_dense1_b), 0)
        x = np.maximum((np.dot(x, weights_dense2_w) + weights_dense2_b), 0)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x


# cartpole test
def test_cartpole(show_plot=False):
    problem = ctsb.problem("Cartpole-v0")
    obs = problem.initialize(render=True)

    model = SmallReactivePolicy()
    model.initialize(problem.get_observation_space(), problem.get_action_space())

    t_start = time.time()
    save_to_mem_ID = -1
    if show_plot:
        frame = 0
        score = 0
        restart_delay = 0
        while time.time() - t_start < 1:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = problem.step(a)

            score += r
            frame += 1
            print("frame: " + str(frame))

    print("about to save to memory")
    save_to_mem_ID = problem.saveToMemory()
    print("save_to_mem_ID: " + str(save_to_mem_ID))

    # run simulator for 4 seconds
    simulator = SimulatorWrapper(problem, save_to_mem_ID)
    simulator.loadFromMemory(simulator.state_id)
    print("simulator.loadFromMemory worked")
    sim_score = score
    sim_frame = frame
    if show_plot:
        while time.time() - t_start < 5:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = simulator.step(a)
            sim_score += r
            sim_frame += 1

    # resume stepping through problem for 2 seconds from the point when the simulator was launched (i.e. t = 1)
    problem.loadFromMemory(save_to_mem_ID)
    print("problem.loadFromMemory worked")
    if show_plot:
        while time.time() - t_start < 7:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = problem.step(a)
            score += r
            frame += 1
            print("frame: " + str(frame))

    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(show_plot=True)

