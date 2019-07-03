
#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import pybullet_envs
import time
import ctsb
from ctsb.models.control.small_reactive_policy import SmallReactivePolicy
from ctsb.problems.control.pendulum import InvertedPendulumSwingupBulletEnv

def main():
  # env = gym.make("InvertedPendulumSwingupBulletEnv-v0")
  # env.render(mode="human")

  # problem = InvertedPendulumSwingupBulletEnv()
  problem = ctsb.problem("Pendulum-v0")
  obs = problem.initialize()

  # pi = SmallReactivePolicy(env.observation_space, env.action_space)

  model = SmallReactivePolicy()
  model.initialize(problem.get_observation_space(), problem.get_action_space())

  while 1:
    frame = 0
    score = 0
    restart_delay = 0
    # obs = env.reset()

    while 1:
      time.sleep(1. / 60.)
      # a = pi.act(obs)
      # obs, r, done, _ = env.step(a)

      a = model.predict(obs)
      obs, r, done, _ = problem.step(a)

      score += r
      frame += 1
      # still_open = env.render("human")
      still_open = problem.render("human")
      if still_open == False:
        return
      if not done: continue
      if restart_delay == 0:
        print("score=%0.2f in %i frames" % (score, frame))
        restart_delay = 60 * 2  # 2 sec at 60 fps
      else:
        restart_delay -= 1
        if restart_delay > 0: continue
        break

if __name__ == "__main__":
  main()


