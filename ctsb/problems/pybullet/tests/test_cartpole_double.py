"""
Test for PyBullet cartpole problem
"""
import time
import tigercontrol
import jax.numpy as np
import jax.random as random
from tigercontrol.models.control.control_model import ControlModel
from tigercontrol.utils import generate_key

# neural network policy class trained specifically for the cartpole problem
class SmallReactivePolicy(ControlModel):
    "Simple multi-layer perceptron policy, no internal state"

    def __init__(self):
        self.initialized = False

    def initialize(self, observation_space, action_space):
        self.initialized = True
        self.weights_dense_w = random.normal(generate_key(), shape=(action_space[0], observation_space[0]))

    def predict(self, ob): # weights can be fount at the end of the file
        return np.dot(self.weights_dense_w, ob)



# cartpole test
def test_cartpole_double(verbose=False):
    # try to break this test
    problem = tigercontrol.problem("PyBullet-CartPoleSwingup-v0")
    obs = problem.initialize(render=False)
    #problem.close()

    problem = tigercontrol.problem("PyBullet-CartPoleDouble-v0")
    obs = problem.initialize(render=verbose)

    model = SmallReactivePolicy()
    model.initialize(problem.get_observation_space(), problem.get_action_space())

    t_start = time.time()
    save_to_mem_ID = -1

    frame = 0
    score = 0
    restart_delay = 0
    saved = False
    while time.time() - t_start < 3:
        time.sleep(1. / 60.)
        a = model.predict(obs)
        obs, r, done, _ = problem.step(a)

        score += r
        frame += 1
        if time.time() - t_start > 0 and not saved:
            if verbose:
                print("about to save to memory")
            #save_to_mem_ID = problem.getState()
            saved = True
        if not done: continue
        if restart_delay == 0:
            if verbose:
                print("score=%0.2f in %i frames" % (score, frame))
            restart_delay = 60 * 2  # 2 sec at 60 fps
        else:
            restart_delay -= 1
            if restart_delay > 0: continue
            break

    problem.close()
    print("test_cartpole_double passed")


if __name__ == "__main__":
    test_cartpole_double(verbose=True)

