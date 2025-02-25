# DQN Charging Pricing Project

This project demonstrates the use of stable-baselines3 to implement a DQN algorithm for optimizing charging pricing in a multi-region charging station environment.

## Project Structure

- **charging_env.py**: Defines the custom OpenAI Gym environment `ChargingEnv`.  
  - The environment includes a state space with various charging demand indicators, previous price, competitor price, and cumulative revenue.
  - It uses a discrete action space for selecting charging prices.
  - The reward function is designed with immediate and terminal rewards.

- **dqn_agent.py**: Implements the DQN agent using stable-baselines3.  
  - It encapsulates the training process, model saving, and loading.

- **main.py**: Entry point of the project.  
  - Creates the environment, verifies it with Gym's interface standards, initializes the DQN agent, and starts training.
  - Saves the trained model to disk.

- **requirements.txt**: Lists the required Python packages for the project.

## Requirements

- Python 3.6+
- `gym`
- `numpy`
- `stable-baselines3`

## Setup and Usage

1. **Install Dependencies**

   Install the required packages by running:
   ```bash
   pip install -r requirements.txt
