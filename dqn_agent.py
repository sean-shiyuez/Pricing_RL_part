import gym
from stable_baselines3 import DQN


class DQNAgent:
    """
    DQN Agent module that encapsulates the training process using stable_baselines3.DQN.
    """

    def __init__(self, env: gym.Env,
                 learning_rate=1e-3,
                 buffer_size=50000,
                 learning_starts=1000,
                 batch_size=32,
                 gamma=0.99,
                 target_update_interval=1000):
        self.env = env
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            verbose=1
        )

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, load_path):
        self.model = DQN.load(load_path, env=self.env)
