from functions import *


coordinates = create_coordinates(plot=False, num_entries=None)
distance_matrix = shortest_path(num_entries=None)
passengers= passenger_demand(num_entries=None)

num_nodes = len(coordinates['GeoUID'])
num_vehicles = 100
num_timesteps = 96
epsilon = 0.1

data = create_parameters(distance_matrix, num_vehicles, num_nodes, num_timesteps, print_values=True)
model = solve_opt(data,pass_min=2000, mult_2=0,tee=True)

# Plot the results
D_passenger = (data['Total demand'][:96]).tolist()
O_energy = (data['Outage'][:96]).tolist()
plot_variables(model, D_passenger, O_energy)

# Save results to Excel
filename = 'test.xlsx'
save_to_excel(model, filename=filename)