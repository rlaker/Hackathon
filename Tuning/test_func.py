import sys

sys.path.append("../")
import multiprocessing as mp
import time as time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from Hack import load, rl


def objective(idx):
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

    # this part is the slow part that we want to split across processors:

    mean_reward_eval = rl.quick_eval(model)
    return mean_reward_eval


if __name__ == "__main__":
    time_start = time.time()
    with mp.Pool(5) as p:
        val_list = p.map(objective, range(10))
    print("mean = ", np.mean(np.array(val_list)))
    time_stop = time.time()
    print("time = ", time_stop - time_start)
    print("val_list = ", val_list)
