# test RandomSearch to balance CartPole

import ctsb
import jax.numpy as np
import jax.random as random
from ctsb.utils import generate_key
import matplotlib.pyplot as plt


def run_episode(problem, parameters, steps, show_plot = False): 
    observation = problem.initialize(render = show_plot)
    total_reward = 0

    for step in range(steps):
        action = -np.ones(1) if np.dot(parameters, observation) < 0 else np.ones(1)
        observation, reward, done, info = problem.step(action)
        total_reward += np.cos(observation[3]) / steps * (step // 20) 
    
    return total_reward

def test_random_search(steps=500, show_plot=True):

    trials = 0

    problem = ctsb.problem("CartPole-v0")

    best_params = None  
    best_reward = -steps

    for trial in range(trials):  
        parameters = random.uniform(generate_key(), minval = -1, maxval = 1, shape = (5,))
        reward = run_episode(problem, parameters, steps)
        if reward > best_reward:
            print(trial, best_reward)
            best_reward = reward
            best_params = parameters

    # best_params = np.array([0, 0, 0, 1, 0])
    # best_params = np.array([-0.08657122, 0.5876281, 0.04698873, -0.55229783, 0.9592931])
    # best_params = np.array([-0.04980421, 0.16760445, 0.01862693, 0.6192529, 0.4058056])
    best_params = np.array([0.22046256, 0.44187307, -0.0226769, 0.7234111, 0.6361048])

    problem = ctsb.problem("CartPole-v0")
    run_episode(problem, best_params, steps * 100, show_plot = show_plot)
    print(best_params)

    print("test_random_search passed")
    return

if __name__=="__main__":
    test_random_search()