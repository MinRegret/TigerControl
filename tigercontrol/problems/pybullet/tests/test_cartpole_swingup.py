"""
Test for PyBullet cartpole problem
"""
import time
import tigercontrol
import jax.numpy as np

# cartpole test
def test_cartpole_swingup(verbose=False):
    problem = tigercontrol.problem("PyBullet-CartPoleSwingup-v0")
    obs = problem.initialize(render=verbose)

    model = tigercontrol.model("CartPoleNN")
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
    print("test_cartpole_swingup passed")


if __name__ == "__main__":
    test_cartpole_swingup(verbose=True)

