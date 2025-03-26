import gym
from gym import spaces
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import beta
from parameters import *

class ChargingStationEnv(gym.Env):
    def __init__(self):
        super(ChargingStationEnv, self).__init__()

        # Import parameters from parameters.py
        self.map_side_size = map_side_size
        self.total_epochs = total_epochs
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        self.mean_vot = mean_vot
        self.std_dev_vot = std_dev_vot
        self.link_detour_time_minutes = link_detour_time_minutes
        self.U_bar = U_bar
        self.Station_specs = Station_specs

        # Define action space: 10 discrete price options (0.1 to 1.0 USD/KWh)
        self.n_prices = 10
        self.action_space = spaces.Discrete(self.n_prices)
        self.prices = np.linspace(0.1, 1.0, self.n_prices)

        # Define observation space
        self.n_regions = sum(self.Station_specs['competitor'] == False)  # Number of our charging stations
        self.total_time_slots = 3  # Number of occupancy time slots for each charging station
        self.state_size = self.n_regions * self.total_time_slots + 3  # State includes occupancy, previous price, competitor price, current total revenue
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        # Initialize environment attributes
        self.current_step = 0
        self.occupancy_tensor = None
        self.our_price_tensor = None
        self.our_previous_prices = None
        self.copetitors_prices = None
        self.total_revenue_tensor = None
        self.our_revenue_tensor = None
        self.demand_tensor = None
        self.G = None
        self.neighbors_dict = None
        self.num_nodes = None

    def reset(self):
        """Reset environment to initial state"""
        self.env_init()  # Initialize environment, including self.num_nodes and other tensors
        self.current_step = 0

        # Get occupancy of our charging stations
        our_stations = self.Station_specs[self.Station_specs['competitor'] == False]['id'].tolist()
        occupancy_our = self.occupancy_tensor.loc[our_stations].values.flatten()

        # Initial previous period price
        p_previous = 0.0

        # Average competitor price
        p_com = self.Station_specs[self.Station_specs['competitor'] == True]['$/KWh'].mean()

        # Initial total revenue
        R_now = 0.0

        # Combine into state vector
        self.state = np.concatenate([occupancy_our, [p_previous, p_com, R_now]])
        return self.state

    def step(self, action):
        """Execute one environment transition"""
        our_price = self.prices[action]
        self.transition(self.current_step, our_price)

        our_stations = self.Station_specs[self.Station_specs['competitor'] == False]['id'].tolist()
        occupancy_our = self.occupancy_tensor.loc[our_stations].values.flatten()
        p_previous = our_price
        p_com = self.copetitors_prices.mean()
        R_now = self.total_revenue_tensor.sum().sum()
        self.state = np.concatenate([occupancy_our, [p_previous, p_com, R_now]])

        ki = 0.15
        sum_o = occupancy_our.sum()
        Ri = ki * sum_o * our_price
        kt = 0.5
        Rt = kt * R_now if self.current_step == self.total_epochs - 1 else 0.0
        reward = 0.01*(Ri + Rt)

        # print(f"reward:{reward}")

        self.current_step += 1
        done = self.current_step >= self.total_epochs
        return self.state, reward, done, {}

    def render(self, mode='human'):
        """Render environment state (optional)"""
        print(f"Step: {self.current_step}, State: {self.state}")

    def env_init(self):
        # Create bidirectional grid graph
        self.G = self.create_bidirectional_grid_graph(self.map_side_size, self.map_side_size)
        self.num_nodes = self.G.number_of_nodes()  # Set self.num_nodes
        self.neighbors_dict = {node: list(self.G.neighbors(node)) for node in self.G.nodes()}

        # Initialize demand_tensor
        self.demand_tensor = np.zeros((self.num_nodes, self.total_epochs, self.total_time_slots))

        try:
            # Read data file
            MI_flow_data = pd.read_csv("MI_JAN_2023 (TMAS).txt", delimiter="|")

            # Filter specific station_id
            city_centers_station_id = [829499, 829959, 829109, 229109, 829419, 829489, 339029, 339020, 339040, 419769,
                                       419729]
            city_flow_data = MI_flow_data[
                MI_flow_data['station_id'].astype(str).isin([str(id) for id in city_centers_station_id])]

            # Extract hourly flow data
            flow_columns = [f'hour_{i:02d}' for i in range(24)]
            flow_data = city_flow_data[flow_columns]

            # Convert to float and handle outliers
            flow_data = flow_data.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Sample specified number of nodes
            self.flow_data = flow_data.sample(n=self.num_nodes)

        except FileNotFoundError:
            # If file doesn't exist, generate random data
            self.flow_data = pd.DataFrame(np.random.randint(50, 500, size=(self.num_nodes, 24)),
                                          columns=[f'hour_{i:02d}' for i in range(24)])

        # Fill demand_tensor
        for node in range(self.num_nodes):
            ratios = self.compute_ratios(self.alpha_param, self.beta_param)
            traffic_profile = self.flow_data.iloc[node].values
            sigma = 10
            traffic_profile = traffic_profile + np.random.normal(0, sigma, size=len(traffic_profile))
            traffic_profile = np.maximum(traffic_profile, 0)
            self.demand_tensor[node, :, 0] = (traffic_profile * ratios[0]).round()
            self.demand_tensor[node, :, 1] = (traffic_profile * ratios[1]).round()
            self.demand_tensor[node, :, 2] = (traffic_profile * ratios[2]).round()

        # Initialize other tensors
        our_total_stations = sum(self.Station_specs['competitor'] == False)
        self.our_price_tensor = pd.DataFrame(
            np.zeros((our_total_stations, self.total_epochs)),
            index=self.Station_specs.loc[self.Station_specs['competitor'] == False, 'id']
        )
        self.occupancy_tensor = pd.DataFrame(
            np.zeros((len(self.Station_specs), self.total_time_slots)),
            index=self.Station_specs['id'],
            columns=['time_slot' + str(x + 1) for x in range(self.total_time_slots)]
        ).rename_axis("sta_id")
        self.our_revenue_tensor = pd.DataFrame(
            np.zeros((our_total_stations, self.total_epochs)),
            index=self.Station_specs.loc[self.Station_specs['competitor'] == False, 'id']
        )
        self.total_revenue_tensor = pd.DataFrame(
            np.zeros((our_total_stations, 1)),
            index=self.Station_specs.loc[self.Station_specs['competitor'] == False, 'id']
        )

    def transition(self, hour, our_price):
        self.our_price_tensor.iloc[:, hour] = our_price
        new_occupancy_tensor = self.occupancy_tensor.copy()
        for sta_id in self.Station_specs['id']:
            if self.Station_specs.loc[self.Station_specs['id'] == sta_id, 'competitor'].values[0]:
                price = self.Station_specs.loc[self.Station_specs['id'] == sta_id, '$/KWh'].values[0]
            else:
                price = our_price
            node = self.Station_specs.loc[self.Station_specs['id'] == sta_id, 'node'].values[0]
            demand = self.demand_tensor[node, hour, :].sum()
            capacity = self.Station_specs.loc[self.Station_specs['id'] == sta_id, 'capacity'].values[0]
            occupancy = min(demand * (1 - price), capacity)
            new_occupancy_tensor.loc[sta_id, :] = occupancy / self.total_time_slots

        # Update revenue, ensuring consistency with reward
        revenue = (new_occupancy_tensor.loc[self.our_price_tensor.index] * our_price).sum(axis=1)
        self.our_revenue_tensor.iloc[:, hour] = revenue
        self.occupancy_tensor = new_occupancy_tensor
        self.total_revenue_tensor.iloc[:, 0] = self.our_revenue_tensor.sum(axis=1)  # Accumulate revenue across all hours
        self.our_previous_prices = pd.DataFrame(
            np.zeros((sum(self.Station_specs['competitor'] == False), 1)),
            index=self.Station_specs.loc[self.Station_specs['competitor'] == False, 'id']
        )
        if hour > 0:
            self.our_previous_prices.iloc[:, 0] = self.our_price_tensor.iloc[:, hour - 1]
        self.copetitors_prices = self.Station_specs.loc[self.Station_specs['competitor'], '$/KWh']

    # Helper functions
    def create_bidirectional_grid_graph(self, rows, cols):
        """Create bidirectional grid graph"""
        G = nx.DiGraph()
        node_counter = 0
        for r in range(rows):
            for c in range(cols):
                G.add_node(node_counter, pos=(c, -r))
                node_counter += 1
        node_counter = 0
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    G.add_edge(node_counter, node_counter + 1, travel_time=self.link_detour_time_minutes)
                    G.add_edge(node_counter + 1, node_counter, travel_time=self.link_detour_time_minutes)
                if r + 1 < rows:
                    G.add_edge(node_counter, node_counter + cols, travel_time=self.link_detour_time_minutes)
                    G.add_edge(node_counter + cols, node_counter, travel_time=self.link_detour_time_minutes)
                node_counter += 1
        return G

    def compute_ratios(self, alpha_param, beta_param):
        """Compute time slot ratios using Beta distribution"""
        beta_samples = beta.rvs(alpha_param, beta_param, size=1000) * 3
        bin_edges = [0, 1, 2, 3]
        bin_counts, _ = np.histogram(beta_samples, bins=bin_edges)
        total_samples = len(beta_samples)
        ratios = bin_counts / total_samples
        return ratios