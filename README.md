# Charging Station Pricing Optimization

This project implements a reinforcement learning-based pricing optimization system for electric vehicle charging stations. The system uses the PPO (Proximal Policy Optimization) algorithm to learn optimal pricing strategies for multiple charging stations in a competitive market.

## Features

- Multi-station pricing optimization
- Realistic demand modeling using traffic flow data
- Competitor price consideration
- Time-varying demand patterns
- Revenue maximization objective

## Project Structure

```
.
├── main.py              # Main training and testing script
├── environment.py       # Charging station environment implementation
├── parameters.py        # Configuration parameters
├── test.py             # Environment testing script
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python main.py
```

The training process will:
- Initialize the charging station environment
- Train the PPO model
- Save the trained model
- Display training progress and metrics
- Generate training curves

### Testing

To test a trained model:
1. Set `mode = 'test'` in `main.py`
2. Run the script:
```bash
python main.py
```

## Environment Configuration

The environment can be configured through `parameters.py`, which includes:
- Map size and time parameters
- Beta distribution parameters for demand modeling
- User time value distribution parameters
- Utility parameters
- Charging station specifications

## Model Architecture

The system uses:
- PPO algorithm for policy optimization
- Multi-discrete action space for station-specific pricing
- State space including:
  - Station occupancy
  - Previous prices
  - Competitor prices
  - Current revenue

## Results

Training results are saved in:
- Model checkpoints: `ppo_charging_station`
- TensorBoard logs: `./ppo_tensorboard/`

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here] 