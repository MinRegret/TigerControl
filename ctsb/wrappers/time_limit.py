import ctsb


class TimeLimit(ctsb.Wrapper):
    def __init__(self, problem, max_episode_steps=None):
        super(TimeLimit, self).__init__(problem)
        if max_episode_steps is None:
            max_episode_steps = problem.spec.max_episode_steps
        self.problem.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call problem.step() before calling reset()"
        observation, reward, done, info = self.problem.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.problem.reset(**kwargs)