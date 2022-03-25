import sys

sys.path.append("../")

import time as time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from Hack import load, rl


def objective(num_episodes=100):
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
    model.learn(100)
    val_list = []

    for i in range(num_episodes):
        val = rl.quick_eval(i, model)
        val_list.append(val)

    return np.mean(val_list)


if __name__ == "__main__":
    time_start = time.time()
    a = objective(num_episodes=150)
    time_stop = time.time()
    print(a)
    print("time = ", time_stop - time_start)
