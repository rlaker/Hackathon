import sys
sys.path.append("../")
import multiprocessing as mp
import time as time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from Hack import load, rl
import gym

def quick_eval(idx, model):
    """
    Evaluation func for the multiprocessing that we have designed to be as quick as possible!
    """
    print("called")
    env = model.get_env()
    env.reset()
    done = False
    episode_rewards = []
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    return sum(episode_rewards)

def add_two(idx, a, b):
    return (a+b)

def objective():
    """
    Function to take in hyperparameters, train a reinforcement model, and output the "profit" of the model as a metric
    """
    epex = load.epex().load()
    price_array = epex["apx_da_hourly"].values

    start_idx = 0
    end_idx = 4 * 2 * 24 * 7  # start_of_2020 # 2019->2020 # 2*24*7
    obs_price_array = price_array[start_idx:end_idx]

    power = 0.5
    env = rl.energy_price_env(obs_price_array, window_size=24 * 2, power=power)
    model = PPO(MlpPolicy, env, verbose=0)

    starmap_obj2 = [(i, model) for i in range(10)]

    starmap_obj = [(0, i, j) for i, j  in list(zip(range(10), range(10)))]
    # # this part is the slow part that we want to split across processors:
    with mp.Pool(2) as p:
        val_list = p.starmap(add_two, starmap_obj)

    # mean_reward_eval = np.mean(np.array(val_list))
    # return mean_reward_eval
    return sum(val_list), starmap_obj2

if __name__=="__main__":
    a = objective()
    print(a)