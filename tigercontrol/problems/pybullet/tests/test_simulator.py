"""
Test for PyBullet cartpole problem
"""
import time
import tigercontrol
import jax.numpy as np
from tigercontrol.problems.pybullet.simulator import Simulator

# cartpole test
def test_simulator(verbose=False):
    problem = tigercontrol.problem("PyBullet-CartPole-v0")
    obs = problem.initialize(render=verbose)

    model = tigercontrol.model("CartPoleNN")
    model.initialize(problem.get_observation_space(), problem.get_action_space())

    t_start = time.time()
    save_to_mem_ID = -1

    frame = 0
    score = 0
    restart_delay = 0
    while time.time() - t_start < 3:
        a = model.predict(obs)
        obs, r, done, _ = problem.step(a)
        score += r
        frame += 1
        if verbose:
            time.sleep(1. / 60.)

    if verbose:
        print("about to save state")
    save_to_mem_ID = problem.getState()
    if verbose:
        print("save_state_ID: " + str(save_to_mem_ID))

    # run simulator for 4 seconds
    problem.loadState(problem.getState())
    sim = problem.fork()

    if verbose:
        print("problem.loadState worked")
    sim_score = score
    sim_frame = frame
    while time.time() - t_start < 3:
        if verbose:
            time.sleep(1. / 60.)
        a = model.predict(obs)
        obs, r, done, _ = problem.step(a)
        sim_score += r
        sim_frame += 1

    # resume stepping through problem for 2 seconds from the point when the simulator was launched (i.e. t = 1)
    problem.loadState(save_to_mem_ID)
    if verbose:
        print("problem.loadState worked")
    while time.time() - t_start < 3:
        a = model.predict(obs)
        obs, r, done, _ = problem.step(a)
        score += r
        frame += 1
        if verbose:
            time.sleep(1. / 60.)

    problem.close()
    print("test_simulator passed")


if __name__ == "__main__":
    test_simulator(verbose=True)

