"""
Test for PyBullet cartpole problem
"""
import time
import ctsb
import jax.numpy as np
from ctsb.models.control.cartpole_nn import CartpoleNN


# cartpole test
def test_cartpole_swingup(show_plot=False):
    problem = ctsb.problem("CartPoleSwingup-v0")
    obs = problem.initialize(render=show_plot)

    model = ctsb.model('CartpoleNN')
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
                save_to_mem_ID = problem.getState()
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
    problem.loadState(save_to_mem_ID)
    print("loadState worked")
    if show_plot:
        while time.time() - t_start < 6:
            time.sleep(1. / 60.)
            a = model.predict(obs)
            obs, r, done, _ = problem.step(a)
            score += r
            frame += 1

    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole_swingup(show_plot=True)

