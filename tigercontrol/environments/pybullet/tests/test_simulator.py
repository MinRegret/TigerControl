"""
Test for PyBullet cartpole environment
"""
import time
import tigercontrol
import jax.numpy as np
from tigercontrol.environments.pybullet.simulator import Simulator

# cartpole test
def test_simulator(verbose=False):
    environment = tigercontrol.environment("PyBullet-CartPole-v0")
    obs = environment.initialize(render=verbose)

    method = tigercontrol.method("CartPoleNN")
    method.initialize(environment.get_observation_space(), environment.get_action_space())

    t_start = time.time()
    save_to_mem_ID = -1

    frame = 0
    score = 0
    restart_delay = 0
    while time.time() - t_start < 3:
        a = method.predict(obs)
        obs, r, done, _ = environment.step(a)
        score += r
        frame += 1
        if verbose:
            time.sleep(1. / 60.)

    if verbose:
        print("about to save state")
    save_to_mem_ID = environment.getState()
    if verbose:
        print("save_state_ID: " + str(save_to_mem_ID))

    # run simulator for 4 seconds
    environment.loadState(environment.getState())
    sim = environment.fork()

    if verbose:
        print("environment.loadState worked")
    sim_score = score
    sim_frame = frame
    while time.time() - t_start < 3:
        if verbose:
            time.sleep(1. / 60.)
        a = method.predict(obs)
        obs, r, done, _ = environment.step(a)
        sim_score += r
        sim_frame += 1

    # resume stepping through environment for 2 seconds from the point when the simulator was launched (i.e. t = 1)
    environment.loadState(save_to_mem_ID)
    if verbose:
        print("environment.loadState worked")
    while time.time() - t_start < 3:
        a = method.predict(obs)
        obs, r, done, _ = environment.step(a)
        score += r
        frame += 1
        if verbose:
            time.sleep(1. / 60.)

    environment.close()
    print("test_simulator passed")


if __name__ == "__main__":
    test_simulator(verbose=True)

