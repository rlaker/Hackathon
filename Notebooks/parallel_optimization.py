import sys

sys.path.append("../")

import optuna
from pathos.multiprocessing import ProcessingPool
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from Hack import load, rl


def create_train_model(price_array, learning_rate, power=0.5, window_size=2 * 24):
    """
    Creates and trains the deep learning model
    """
    start_idx = 0
    end_idx = 4 * 2 * 24 * 7  # start_of_2020 # 2019->2020 # 2*24*7
    obs_price_array = price_array[start_idx:end_idx]
    env = rl.energy_price_env(obs_price_array, window_size=window_size, power=power)
    model = PPO(MlpPolicy, env, verbose=0, learning_rate=learning_rate)
    model.learn(100)
    return model


def evaluate_model(iter_obj):
    total_val = 0
    for obj in iter_obj:
        idx = obj[0]
        model = obj[1]
        val = rl.quick_eval(idx, model)
        total_val += val
    return total_val


def objective(trial):
    """
    args: params of our model
    """
    data = load.epex().load()
    price_array = data["apx_da_hourly"].values
    learning_rate = trial.suggest_float("learning_rate", 1.0e-5, 1.0e-3)
    model = create_train_model(price_array, learning_rate)
    obj_to_iterate = [(i, model) for i in range(10)]
    obj_to_iterate2 = [(i, model) for i in range(10)]
    obj_to_iterate3 = [(i, model) for i in range(10)]
    results = ProcessingPool(3).map(
        evaluate_model, [obj_to_iterate, obj_to_iterate2, obj_to_iterate3]
    )
    # calculate the mean result
    mean_result = sum(results) / (
        float(len(obj_to_iterate) + len(obj_to_iterate2) + len(obj_to_iterate3))
    )
    return mean_result[0]


if __name__ == "__main__":
    # carry out the optimisation part of the problem here
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
