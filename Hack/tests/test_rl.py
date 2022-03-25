import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from Hack import rl


@pytest.fixture
def price_array():
    return np.array([10, 20, 30, 20, 10])


@pytest.fixture
def env(price_array):
    return rl.energy_price_env(price_array, window_size=3, power=1)


@pytest.mark.parametrize("human_action, mapped_action", [(-1, 0), (0, 1), (1, 2)])
def test_mapped_actions(human_action, mapped_action):
    assert mapped_action == rl.human2env(human_action)
    assert human_action == rl.env2human(mapped_action)


def test_get_mode():

    # just give it lots of ones and see if it gives you one
    data = np.ones(100)

    mode = rl.get_mode(data, 10)

    assert mode == pytest.approx(1, 0.2)
    mode = rl.get_mode(np.array([np.nan]), 10)
    assert np.isnan(mode)


@pytest.mark.parametrize(
    "idx, expected_price", [(0, 10), (1, 15), (2, 20), (3, 20), (4, 20)]
)
def test_expected_price_median(price_array, idx, expected_price):
    assert expected_price == rl.get_expected_price(price_array, idx, 3, "median")


def test_env(env):
    # this should output a None if it passes
    assert check_env(env, warn=True) is None


def test_sell(env):
    expected_states = [
        np.array([20, 15, 0.5, 1]),
        np.array([30, 20, 0, 2]),
        np.array([20, 20, 0, 3]),
        np.array([10, 20, 0, 4]),
    ]
    for i, expected_state in enumerate(expected_states):
        state, reward, done, _ = env.step(rl.human2env(-1))
        assert np.allclose(state, expected_state)
        if i == len(expected_states) - 1:
            assert done


def test_buy(price_array):
    env = rl.energy_price_env(price_array, start_energy=0)
    expected_states = [
        np.array([20, 15, 0.425, 1]),
        np.array([30, 20, 0.85, 2]),
        np.array([20, 20, 1, 3]),
        np.array([10, 20, 1, 4]),
    ]

    for i, expected_state in enumerate(expected_states):
        state, reward, done, _ = env.step(rl.human2env(1))
        assert np.allclose(state, expected_state)
        if i == len(expected_states) - 1:
            assert done


def test_hold(env):
    expected_states = [
        np.array([20, 15, 1, 1]),
        np.array([30, 20, 1, 2]),
        np.array([20, 20, 1, 3]),
        np.array([10, 20, 1, 4]),
    ]

    for i, expected_state in enumerate(expected_states):
        state, reward, done, _ = env.step(rl.human2env(0))
        assert np.allclose(state, expected_state)
        if i == len(expected_states) - 1:
            assert done


@pytest.mark.parametrize(
    "human_action, current_energy, new_energy",
    [
        (-1, 1, 0.5),
        (0, 1, 1),
        (1, 0, 0.425),
    ],
)
def test_apply_action(env, human_action, current_energy, new_energy):
    assert env.apply_action(human_action, current_energy) == new_energy


@pytest.mark.parametrize(
    "delta_energy, current_price, expected_price, expected_reward",
    [
        (1, 10, 10, 0),
        (1, 20, 10, -10),
        (-1, 20, 10, 10),
        (1, 10, 20, 10),
        (-1, 10, 20, -10),
    ],
)
def test_reward(env, delta_energy, current_price, expected_price, expected_reward):
    assert (
        env.get_reward(delta_energy, current_price, expected_price) == expected_reward
    )
