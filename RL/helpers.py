import gym
import matplotlib.pyplot as plt
import numpy as np


class energy_price_env(gym.Env):
    def __init__(self, price_array, start_energy=1, start_time=0, max_time=7 * 24 * 2):
        self.price_array = price_array
        self.action_space = gym.spaces.Discrete(3)
        # current_price, mean_price, current_energy, time
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, 0]),
            high=np.array([np.inf, np.inf, 1, np.inf]),
            dtype=np.float32,
        )
        # our state is the charge
        self.start_energy = start_energy
        self.state = np.array(
            [self.get_price(start_time), self.get_mean_price(start_time), start_energy]
        )
        self.start_time = start_time
        self.earnings = 0
        self.power = 1  # MW
        self.capacity = 1  # MWh
        self.efficiency = 0.85
        self.max_time = max_time

    def get_state(self):
        return self.state

    def get_price(self, idx):
        return self.price_array[int(idx)]

    def get_mean_price(self, idx):
        idx = int(idx)
        window_size = 1000
        if idx == 0:
            return self.price_array[idx]
        elif idx < window_size:
            return np.mean(self.price_array[:idx])
        else:
            return np.mean(self.price_array[idx - window_size : idx])

    def step(self, action):
        current_price, mean_price, current_energy, current_time = self.state
        mapped_action = env2human(action)
        if mapped_action == -1:
            # discharge === selling for 30 mins
            new_energy = current_energy - (self.power * 0.5)

        elif mapped_action == 0:
            # hold === do nothing
            new_energy = current_energy
        elif mapped_action == 1:
            # charge === buy energy for 30 mins
            new_energy = current_energy + (self.power * 0.5 * self.efficiency)

        # make sure energy cannot be greater than capacity
        new_energy = max(0, new_energy)
        new_energy = min(self.capacity, new_energy)
        # now work out the delta energy
        delta_energy = new_energy - current_energy

        revenue = -delta_energy * current_price
        self.earnings += revenue
        expected_profit = -delta_energy * mean_price

        opportunity_cost = revenue - expected_profit

        reward = opportunity_cost  # profit * multiplier * price_diff_from_expected

        # print("Delta energy: ", delta_energy)
        # print("Price diff from expected: ", price_diff_from_expected)
        # print("Revenue: ", revenue)
        # print("Expected Profit: ", expected_profit)
        # print("Reward ", reward)

        # increase the time
        current_time += 1

        self.state = (
            self.get_price(current_time),
            self.get_mean_price(current_time),
            new_energy,
            current_time,
        )

        info = {}
        if current_time >= self.max_time + self.start_time:
            done = True
        else:
            done = False

        return np.array(self.state), reward, done, info

    def reset(self):
        # this resets the environment so it can try again
        # print('Environment reset')
        self.state = np.array(
            [
                self.get_price(self.start_time),
                self.get_mean_price(self.start_time),
                self.start_energy,
                self.start_time,
            ]
        )
        # ! this puts us at the start of the week, we could make this random?
        self.earnings = 0
        return self.state


def humans2env(action):
    return int(action + 1)


def env2human(action):
    return int(action - 1)


def evaluate(model, new_env=None, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    if new_env is None:
        env = model.get_env()
    else:
        env = new_env
    env.reset()
    all_episode_rewards = []

    for i in range(num_episodes):

        episode_rewards = []
        if i == 0:
            current_prices = []
            mean_prices = []
            current_energies = []
            all_earnings = [0]

        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            current_price, mean_price, current_energy, current_time = (
                obs[0, 0],
                obs[0, 1],
                obs[0, 2],
                obs[0, 3],
            )
            episode_rewards.append(reward)

            if i == 0:
                if len(current_energies) > 0:
                    all_earnings.append(
                        -current_price * (current_energy - current_energies[-1])
                    )

                current_prices.append(current_price)
                mean_prices.append(mean_price)
                current_energies.append(current_energy)

        all_episode_rewards.append(sum(episode_rewards))

    fig, axs = plt.subplots(4, 1, sharex=True)
    index = np.arange(0, len(current_energies))
    cum_rewards = np.cumsum(episode_rewards)
    bank_total = np.cumsum(all_earnings)
    axs[0].plot(index, cum_rewards, color="red", label="Cumalative rewards")
    axs[0].plot(index, bank_total, color="blue", label="Bank total")
    axs[0].legend()
    axs[1].plot(index, current_prices, color="blue", label="Current prices")
    axs[1].plot(index, mean_prices, color="red", label="Mean prices")
    axs[1].legend()

    axs[2].plot(index, episode_rewards, color="black", label="Reward")
    axs[2].legend()

    axs[3].plot(index, current_energies, color="blue", label="Current energies")

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward
