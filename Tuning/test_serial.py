import sys

sys.path.append("../")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from Hack import load, rl

"""
This script is what we want to somehow parallelise for tuning
"""


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
    val_list = []

    for i in range(num_episodes):
        val = rl.quick_eval(i, model)
        val_list.append(val)

    return np.mean(val_list)


if __name__ == "__main__":
    a = objective()
    print(a)
