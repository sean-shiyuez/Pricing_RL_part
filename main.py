from charging_env import ChargingEnv
from dqn_agent import DQNAgent
from stable_baselines3.common.env_checker import check_env


def main():
    # Create an instance of the charging environment.
    env = ChargingEnv(n_regions=3, n_price_options=5, step_interval=1.0)

    # Optionally check if the environment conforms to Gym's interface standards.
    check_env(env, warn=True)

    # Initialize the DQN agent.
    agent = DQNAgent(env)

    # Train the agent; adjust total_timesteps as needed.
    agent.train(total_timesteps=10000)

    # Save the trained model.
    agent.save("dqn_charging_model")


if __name__ == "__main__":
    main()
