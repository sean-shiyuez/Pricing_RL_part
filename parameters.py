import pandas as pd

# Map and time parameters
map_side_size = 3
total_epochs = 24
link_detour_time_minutes = 5

# Beta distribution parameters
alpha_param = 2
beta_param = 5

# User time value distribution parameters
mean_vot = 25
std_dev_vot = 10

# Utility parameters
U_bar = -6

# Define charging station dataframe
col_names = ['id', 'node', 'competitor', 'capacity', '$/KWh', 'Charging_rate(KW)']
Station_specs = pd.DataFrame(columns=col_names)
Station_specs.loc[len(Station_specs)] = [1, 0, True, 400, 0.412, 12]
Station_specs.loc[len(Station_specs)] = [2, 4, True, 399, 0.4, 15]
Station_specs.loc[len(Station_specs)] = [3, 8, True, 350, 0.423, 18]
Station_specs.loc[len(Station_specs)] = [4, 3, True, 250, 0.22, 16]
Station_specs.loc[len(Station_specs)] = [5, 3, False, 350, "RL", 17]
Station_specs.loc[len(Station_specs)] = [6, 4, False, 320, "RL", 14]
Station_specs.loc[len(Station_specs)] = [7, 2, False, 280, "RL", 16]
