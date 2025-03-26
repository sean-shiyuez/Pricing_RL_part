from pickle import FALSE

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environment import ChargingStationEnv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# np.random.seed(42)
# np.random.seed(41)
# 1 5 4 6

class RevenueEvalCallback(BaseCallback):
    def __init__(self, env, eval_freq=100, verbose=1, tensorboard_log_dir="./ppo_tensorboard/"):
        super(RevenueEvalCallback, self).__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq
        self.episode_reward = 0.0
        self.total_revenues = []
        self.total_incomes = []
        self.eval_rewards = []
        self.tensorboard_log_dir = tensorboard_log_dir

    def _on_step(self):
        self.episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            total_income = self.env.total_revenue_tensor.sum().sum()
            self.total_revenues.append(self.episode_reward)
            self.total_incomes.append(total_income)
            print(f"Episode {len(self.total_revenues)}: Total training reward = {self.episode_reward}, Total income = {total_income}")
            self.episode_reward = 0.0
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=10)
            self.eval_rewards.append(mean_reward)
            print(f"Step {self.n_calls}: Average evaluation reward = {mean_reward}, Standard deviation = {std_reward}")
        return True

    def plot_curves(self):
        plt.figure(figsize=(12, 5))
        loss_values = self._load_tensorboard_loss()
        plt.subplot(1, 2, 1)
        if loss_values:
            plt.plot(loss_values, label='Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
        else:
            print("No loss data found in TensorBoard logs.")
        plt.legend()
        plt.subplot(1, 2, 2)
        if self.eval_rewards:
            eval_steps = [i * self.eval_freq for i in range(1, len(self.eval_rewards) + 1)]
            plt.plot(eval_steps, self.eval_rewards, label='Evaluation Reward')
            plt.xlabel('Steps')
            plt.ylabel('Mean Reward')
            plt.title('Evaluation Reward Curve')
        else:
            print("No evaluation reward data available.")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _load_tensorboard_loss(self):
        if not os.path.exists(self.tensorboard_log_dir):
            print(f"TensorBoard log directory {self.tensorboard_log_dir} does not exist.")
            return []
        log_files = []
        for root, _, files in os.walk(self.tensorboard_log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    log_files.append(os.path.join(root, file))
        if not log_files:
            print("No TensorBoard log files found.")
            return []
        latest_log_file = max(log_files, key=os.path.getctime)
        event_acc = EventAccumulator(latest_log_file)
        event_acc.Reload()
        loss_values = []
        if 'train/loss' in event_acc.Tags()['scalars']:
            for event in event_acc.Scalars('train/loss'):
                loss_values.append(event.value)
        elif 'rollout/loss' in event_acc.Tags()['scalars']:
            for event in event_acc.Scalars('rollout/loss'):
                loss_values.append(event.value)
        else:
            print("No loss data found in TensorBoard logs.")
        return loss_values


def train(env, total_timesteps=10000):
    # Enable TensorBoard logging
    tensorboard_log_dir = "./ppo_tensorboard/"
    # Create PPO model with appropriate hyperparameters for multi-discrete action space
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tensorboard_log_dir
    )
    callback = RevenueEvalCallback(env, eval_freq=100, tensorboard_log_dir=tensorboard_log_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_charging_station")

    # Output total rewards and incomes for all episodes after training
    print("Training completed. Total rewards and incomes for all episodes:")
    for i, (reward, income) in enumerate(zip(callback.total_revenues, callback.total_incomes)):
        print(f"Episode {i + 1}: Total reward = {reward}, Total income = {income}")

    # Plot loss and reward curves
    callback.plot_curves()
    return model

def test_model(env, model_path="ppo_charging_station"):
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    total_revenue = 0
    print("Testing model actions:")
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Use deterministic=True for testing
        obs, reward, done, info = env.step(action)
        total_revenue += reward
        print(f"Action: {action}, Immediate reward: {reward}")
    print(f"Total test reward: {total_revenue}")

def main():
    env = ChargingStationEnv()
    mode = 'train'
    if mode == 'train':
        print("Starting training...")
        train(env, total_timesteps=10000)
    elif mode == 'test':
        print("Starting testing...")
        test_model(env, model_path="ppo_charging_station")
    else:
        print("Invalid mode. Please set mode to 'train' or 'test'")

if __name__ == "__main__":
    main()

