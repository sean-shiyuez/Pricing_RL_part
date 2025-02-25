import gym
import numpy as np
from gym import spaces


class ChargingEnv(gym.Env):
    """
    Charging Environment.

    State space:
      [o13, o12, o11, o23, o22, o21, ..., on3, on2, on1, p_previous, p_com, R_now]
      where o13 represents the user in region i who still needs 3 hours to charge
      (similarly for o12 and o11), p_previous is the previous time step's price,
      p_com is the competitor's charging price, and R_now is the current cumulative revenue.

    Action space:
      Discrete charging price options a = [p1, p2, ..., pj], for example, 5 price options.

    Reward design:
      - Immediate reward: Ri = k_i * (sum of all o values) * current selected price.
      - Terminal reward (at episode end): Rt = k_t * R_now.

    Note:
      The state transition and environment dynamics are provided as a simple example.
      The actual dynamic system (e.g., user inflow/outflow, charging time updates) is the core innovative component,
      to be designed by Iason.
    """

    def __init__(self, n_regions=3, n_price_options=5, step_interval=1.0):
        super(ChargingEnv, self).__init__()
        self.n_regions = n_regions
        self.n_price_options = n_price_options
        self.step_interval = step_interval  # Time interval: 1 hour

        # Reward coefficients
        self.k_i = 1.0  # Immediate reward coefficient
        self.k_t = 1.0  # Terminal reward coefficient

        # Define the state space:
        # Each region has 3 o values, plus p_previous, p_com, and R_now.
        # Total dimensions: n_regions * 3 + 3.
        self.state_dim = self.n_regions * 3 + 3
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.state_dim,), dtype=np.float32)

        # Define the action space:
        # Discrete price selection with n_price_options options,
        # e.g., evenly spaced prices between 1 and 10.
        self.action_space = spaces.Discrete(self.n_price_options)
        self.price_options = np.linspace(1, 10, self.n_price_options)

        # Control the episode length (e.g., 24 steps represent 24 hours).
        self.max_steps = 24

        self.reset()

    def reset(self):
        # Initialize the state: can be random or set based on specific initial conditions.
        self.state = np.random.uniform(low=0, high=10, size=(self.state_dim,))
        self.state[-1] = 0  # Initialize R_now to 0
        self.step_count = 0

        # This part is the core innovative content, designed by Iason.
        return self.state

    def step(self, action):
        # Get the price corresponding to the current action.
        price = self.price_options[action]

        # Compute the immediate reward.
        # The o values are the first n_regions * 3 elements of the state.
        o_values = self.state[:self.n_regions * 3]
        immediate_reward = self.k_i * np.sum(o_values) * price

        # Update cumulative revenue R_now (the last element of the state).
        self.state[-1] += immediate_reward

        # Update the environment state.
        # This is a simplified example. The actual dynamics (e.g., user inflow/outflow, charging duration decrement)
        # should be designed by Iason.
        self.state = self.state + np.random.uniform(-0.5, 0.5, size=self.state.shape)

        # Increase the step count and check if the episode has finished.
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # When the episode ends, add the terminal reward.
        if done:
            terminal_reward = self.k_t * self.state[-1]
            reward = immediate_reward + terminal_reward
        else:
            reward = immediate_reward

        info = {}
        return self.state, reward, done, info
