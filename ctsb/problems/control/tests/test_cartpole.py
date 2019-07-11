"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np
from ctsb.models.control.control_model import ControlModel
from cartpole_weights import *


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
    problem = ctsb.problem("CartPole-v0")
    obs = problem.initialize(render=True)

    model = SmallReactivePolicy()
    model.initialize(problem.get_observation_space(), problem.get_action_space())

    t_start = time.time()
    save_to_mem_ID = -1
    if show_plot:
        frame = 0
        score = 0
        restart_delay = 0
        saved = False
        while time.time() - t_start < 5:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = problem.step(a)

            score += r
            frame += 1
            #still_open = problem.render("human")
            #if still_open == False:
            #    return
            if time.time() - t_start > 0 and not saved:
                print("about to save to memory")
                save_to_mem_ID = problem.saveToMemory()
                saved = True
            if not done: continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay > 0: continue
                break

    print("save_to_mem_ID: " + str(save_to_mem_ID))
    problem.loadFromMemory(save_to_mem_ID)
    print("loadFromMemory worked")
    if show_plot:
        while time.time() - t_start < 6:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = problem.step(a)
            score += r
            frame += 1

    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(show_plot=True)

