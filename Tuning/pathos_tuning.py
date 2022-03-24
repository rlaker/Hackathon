import sys

from pathos.multiprocessing import ProcessingPool

sys.path.append("../")
import time as time

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from Hack import load, rl


def create_train_model(price_array, learning_rate):
    """
    Creates and trains the deep learning model
    """
    start_idx = 0
    end_idx = 4 * 2 * 24 * 7  # start_of_2020 # 2019->2020 # 2*24*7
    obs_price_array = price_array[start_idx:end_idx]
    power = 0.5
    env = rl.energy_price_env(obs_price_array, window_size=24 * 2, power=power)
    model = PPO(MlpPolicy, env, verbose=0, learning_rate=learning_rate)
    return model


def evaluate_model(iter_obj):
    val_list = []
    for obj in iter_obj:
        idx = obj[0]
        model = obj[1]
        val = rl.quick_eval(idx, model)
        val_list.append(val)
    return sum(val_list)


def objective(learning_rate=0.01):
    """
    args: params of our model
    """
    if (
        __name__ == "__main__"
    ):  # only do this calculation if we're on the main processor
        data = load.epex().load()
        price_array = data["apx_da_hourly"].values
        model = create_train_model(price_array, learning_rate)
        obj_to_iterate = [(i, model) for i in range(50)]
        obj_to_iterate2 = [(i, model) for i in range(50)]
        # obj_to_iterate3 = [(i, model) for i in range(100)]
        time_start = time.time()
        results = ProcessingPool(3).map(
            evaluate_model, [obj_to_iterate, obj_to_iterate2]
        )
        # results = ProcessingPool(4).map(evaluate_model, [obj_to_iterate3])
        # calculate the mean result
        time_stop = time.time()
        print("time = ", time_stop - time_start)
        return results
        # mean_result = (results[0] + results[1])/(float(len(obj_to_iterate) + len(obj_to_iterate2)))
        # return mean_result[0]


a = objective()
if a is not None:
    print(a)
