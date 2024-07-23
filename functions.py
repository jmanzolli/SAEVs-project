import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import pandas as pd
import geopandas as gpd

def coordinates(data='/Users/natomanzolli/Documents/PhD/Artigos/Journals/SAEVs paper/Origin_destination_matrices/CT_Centroids_with_YX.csv', plot=True, num_entries=None):
    df = pd.read_csv(data)
    
    # Limit the number of entries if specified
    if num_entries:
        df = df.head(num_entries)

    # Store the coordinates and GeoUID
    coordinates = df[['GeoUID', 'Y', 'X']]

    if plot:
        # Create a GeoDataFrame from the DataFrame
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['X'], df['Y'])
        )

        # Plot the coordinates on a map
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the points
        gdf.plot(ax=ax, color='blue', markersize=5)

        # Optionally, add GeoUID labels to the plot
        for x, y, label in zip(df['X'], df['Y'], df['GeoUID']):
            ax.text(x, y, label, fontsize=8, ha='right', va='bottom')

        # Set plot title and labels
        ax.set_title('Coordinates with GeoUIDs')
        ax.set_xlabel('Longitude (X)')
        ax.set_ylabel('Latitude (Y)')

    plt.show()
    return coordinates

def shortest_path(file_path='/Users/natomanzolli/Documents/PhD/Artigos/Journals/SAEVs paper/Origin_destination_matrices/Shortest_Path_Matrix.csv', num_entries=None):
    # Read the CSV file into a DataFrame, specifying the separator as ';' and setting data types for index and headers
    distances = pd.read_csv(file_path, sep=';', index_col=0, dtype={'origins': str}) / 1000
    
    # Limit the number of entries if specified
    if num_entries:
        distances = distances.head(num_entries)
        distances = distances.loc[:, distances.columns[:num_entries]]
    
    # Convert DataFrame to NumPy array
    distances_array = distances.to_numpy()
    
    return distances_array

def passenger_demand(file_path_shortest_paths='/Users/natomanzolli/Documents/PhD/Artigos/Journals/SAEVs paper/Origin_destination_matrices/Shortest_Path_Matrix.csv', hours_experiment=24, plot=True, num_entries=None):
    # Read the CSV file into a DataFrame, specifying the separator as ';' and setting data types for index and headers
    df_shortest_paths = pd.read_csv(file_path_shortest_paths, sep=';', index_col=0, dtype={'origins': str})
    
    # Limit the number of entries if specified
    if num_entries:
        df_shortest_paths = df_shortest_paths.head(num_entries)
        df_shortest_paths = df_shortest_paths.loc[:, df_shortest_paths.columns[:num_entries]]
    
    # Extract all unique GeoUIDs from the shortest path DataFrame
    geo_uids = df_shortest_paths.index.tolist()
    num_geo_uids = len(geo_uids)

    # Initialize a 3D NumPy array to store hourly passenger demand matrices
    passenger_demand = np.zeros((num_geo_uids, num_geo_uids, hours_experiment))

    # Example of reading passenger demand files for each hour and filling the array
    for hour in range(hours_experiment):
        # Example file path for passenger demand data for each hour
        file_path_demand = f'/Users/natomanzolli/Documents/PhD/Artigos/Journals/SAEVs paper/Origin_destination_matrices/24_1hour_od/car_{hour}h.csv'

        # Read passenger demand data for the current hour
        df_demand_hour = pd.read_csv(file_path_demand, sep=',', index_col=0, dtype={'origins': str})
        
        if num_entries:
            df_demand_hour = df_demand_hour.head(num_entries)
            df_demand_hour = df_demand_hour.loc[:, df_demand_hour.columns[:num_entries]]

        # Fill the array for the current hour
        for origin, row in df_demand_hour.iterrows():
            origin = str(origin)
            if origin.endswith('.0'):
                origin = origin[:-2]  # Remove the '.0' suffix if it exists
            for destination, demand in row.items():
                destination = str(destination)
                if destination.endswith('.0'):
                    destination = destination[:-2]  # Remove the '.0' suffix if it exists
                if pd.notna(demand):
                    if origin in geo_uids and destination in geo_uids:
                        origin_idx = geo_uids.index(origin)
                        destination_idx = geo_uids.index(destination)
                        passenger_demand[origin_idx, destination_idx, hour] = round(demand)

    if plot:
        # Calculate the total number of passengers per hour by summing all non-zero entries
        commutes_per_hour = []

        for hour in range(hours_experiment):
            # Sum all non-zero entries in the array for the current hour
            number_passengers = np.sum(passenger_demand[:, :, hour])
            commutes_per_hour.append(number_passengers)

        # Plotting the number of passengers per hour
        plt.figure(figsize=(10, 4))
        plt.bar(range(hours_experiment), commutes_per_hour, color='blue')
        plt.xlabel('Hour')
        plt.ylabel('Number of Passengers')
        plt.title('Number of Passengers per Hour')
        plt.xticks(range(hours_experiment))  # Set x-axis ticks to match hours
        plt.grid(axis='y')
        plt.show()
    
    return passenger_demand

def create_parameters(distance_matrix, num_vehicles, num_nodes, num_timesteps, print_values=True):
    battery = [100] * num_vehicles  # Battery capacity of vehicle k [kWh]
    pi_plus = np.random.uniform(0.02, 0.05, size=num_timesteps)  # Price to buy energy (grid) at time t [CA$/kWh]
    pi_minus = np.random.uniform(0.01, 0.03, size=num_timesteps)  # Price to sell energy (grid) at time t [CA$/kWh]
    pi_out = np.random.uniform(0.015, 0.035, size=num_timesteps)  # Price to sell energy (outage) at time t [CA$/kWh]
    alpha = 0.5  # Price paid by each customer per trip [CA$/time]
    E_req = np.random.choice([0,100],size=(num_nodes, num_timesteps),p=[0.95,0.05])  # Energy required in an affected zone o per timestep t [kWh]
    gamma_max = 1  # Maximum state-of-charge [%]
    gamma_min = 0.2  # Minimum state-of-charge [%]
    gamma_0 = np.random.uniform(0.75, 1.0, size=num_vehicles)  # Initial state-of-charge for each vehicle k [%]
    # Travel time between node i to node j // average speed: 12 km/h
    tau = np.round((distance_matrix / 12).astype(float))  # Use np.round() for rounding
    tau = np.where(np.isnan(tau) | np.isinf(tau), 0, tau).astype(int)  # Convert to int after handling NaN and Inf
    xi = 2 # Energy consumption rate [kWh/km]
    rho_plus = 10  # Charging rate (parking spots) [kW]
    rho_minus = 10  # Discharging rate (parking spots) [kW]
    rho_out = 5  # Discharging rate (outage location) [kW]
    eta = 0.9  # Charger efficiency [%]
    R_bat = 130  # Battery replacement costs [CA$/kWh]
    N_Cy = 3000  # Number of cycles until end-of-life
    lambda_DoD = 0.8 # Depth of discharge [%]
    
    if print_values == True:
        # Print all parameters
        print(f"Battery capacities (B): {battery}")
        print(f"Price to buy energy (pi_plus): {pi_plus}")
        print(f"Price to sell energy (pi_minus): {pi_minus}")
        print(f"Price to sell energy (outage) (pi_out): {pi_out}")
        print(f"Price paid by each customer (alpha): {alpha}")
        print(f"Energy required (E_req):\n{E_req}")
        print(f"Initial state-of-charge (gamma_0): {gamma_0}")
        print(f"Travel time (tau):\n{tau}")
        print(f"Energy consumption rate (xi): {xi}")
        print(f"Charging rate (parking spots) (rho_plus): {rho_plus}")
        print(f"Discharging rate (parking spots) (rho_minus): {rho_minus}")
        print(f"Discharging rate (outage location) (rho_out): {rho_out}")
        print(f"Charger efficiency (eta): {eta}")
        print(f"Battery replacement costs (R_bat): {R_bat}")
        print(f"Number of cycles until end-of-life (N_Cy): {N_Cy}")
        print(f"Depth of discharge (lambda_DoD): {lambda_DoD}")
    
    return battery, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD

def solveModel(epsilon, battery, passengers, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD, num_nodes, num_vehicles, num_timesteps):
    model = ConcreteModel()
    print('Solving with ɛ =', epsilon)
    
    # Sets
    model.N = RangeSet(num_nodes) # set of nodes
    model.S = RangeSet(num_nodes) # subset of parking spots
    model.O = RangeSet(num_nodes) # subset of outage locations
    model.S_O = RangeSet(num_nodes) # intersection of parking spots and outage
    model.A = Set(dimen=2, initialize=lambda model: [(i, j) for i in model.N for j in model.N if i != j]) # set of arcs
    model.K = RangeSet(num_vehicles) # set of vehicles
    model.T = RangeSet(num_timesteps) # set of timesteps

    # Parameters
    model.B = Param(model.K, initialize=lambda model, k: battery[k-1])  # Battery capacity of vehicle k [kWh]
    model.P = Param(model.N, model.N, model.T, initialize=lambda model, i, j, t: passengers[i-1, j-1, t-1])  # Number of passengers arriving at node i with destination j at time t
    model.pi_plus = Param(model.T, initialize=lambda model, t: pi_plus[t-1])  # Price to buy energy (grid) at time t [CA$/kWh]
    model.pi_minus = Param(model.T, initialize=lambda model, t: pi_minus[t-1])  # Price to sell energy (grid) at time t [CA$/kWh]
    model.pi_out = Param(model.T, initialize=lambda model, t: pi_out[t-1])  # Price to sell energy (outage) at time t [CA$/kWh]
    model.alpha = Param(initialize=alpha)  # Price paid by each customer per trip [CA$/time]
    model.E_req = Param(model.N, model.T, initialize=lambda model, n, t: E_req[n-1, t-1])  # Energy required in an affected zone o per timestep t [kWh]
    model.gamma_max = Param(initialize=gamma_max)  # Maximum state-of-charge [%]
    model.gamma_min = Param(initialize=gamma_min)  # Minimum state-of-charge [%]
    model.gamma_0 = Param(model.K, initialize=lambda model, k: gamma_0[k-1])  # Initial state-of-charge for each vehicle k [%]
    model.tau = Param(model.N, model.N, initialize=lambda model, i, j: tau[i-1, j-1])  # Travel time between node i to node j
    model.xi = Param(initialize=xi)  # Energy consumption rate [kWh/km]
    model.rho_plus = Param(initialize=rho_plus)  # Charging rate (parking spots) [kW]
    model.rho_minus = Param(initialize=rho_minus)  # Discharging rate (parking spots) [kW]
    model.rho_out = Param(initialize=rho_out)  # Discharging rate (outage location) [kW]
    model.eta = Param(initialize=eta)  # Charger efficiency [%]
    model.R_bat = Param(initialize=R_bat)  # Battery replacement costs [CA$/kWh]
    model.N_Cy = Param(initialize=N_Cy)  # Number of cycles until end-of-life
    model.lambda_DoD = Param(initialize=lambda_DoD)  # Depth of discharge [%]

    # Variables
    model.x = Var(model.K, model.N, model.N, model.T, within=Binary)  # Binary variable indicating if vehicle k is carrying passengers from node i to node j at timestep t {0,1}
    model.y = Var(model.K, model.N, model.N, model.T, within=Binary)  # Binary variable indicating if vehicle k is reallocating (empty) from node i to node j at timestep t {0,1}
    model.z = Var(model.K, model.N, model.T, within=Binary)  # Binary variable indicating if vehicle k in parked at node i at timestep t {0,1}
    model.p = Var(model.K, model.N, model.T, within=Binary)  # Binary variable indicating if vehicle k is arriving at node i at timestep t {0,1}
    model.u_plus = Var(model.K, model.S, model.T, within=Binary)  # Binary variable indicating if vehicle k is charging at node s at timestep t {0,1}
    model.u_minus = Var(model.K, model.S, model.T, within=Binary)  # Binary variable indicating if vehicle k is discharging at node s at timestep t {0,1}
    model.u_out = Var(model.K, model.O, model.T, within=Binary)  # Binary variable indicating if vehicle k is offering energy in affected zone o at time t {0,1}
    model.d = Var(model.N, model.N, model.T, within=NonNegativeIntegers)  # Number of passengers waiting at node i with destination j at timestep t
    model.eng = Var(model.K, model.T, within=NonNegativeReals)  # Energy of vehicle k at timestep t [kWh]
    model.g_plus = Var(model.K, model.S, model.T, within=NonNegativeReals)  # Energy charged from vehicle k at node s at timestep t [kWh]
    model.g_minus = Var(model.K, model.S, model.T, within=NonNegativeReals)  # Energy discharged from vehicle k at node s at timestep t [kWh]
    model.g_out = Var(model.K, model.O, model.T, within=NonNegativeReals)  # Total energy offered from vehicle k to affected zone o at timestep t [kWh]
    model.a = Var(model.K, within=NonNegativeReals) # Total energy taken from a vehicle k battery through its lifespan [kWh]
    model.w = Var(model.K, model.T, within=NonNegativeReals)  # Battery degradation costs of vehicle k in timestep t to offer energy to the grid [CA$]

    # Objective 1
    def objective1(model):
        return sum(sum(model.alpha * model.tau[i,j] * model.x[k,i,j,t] for k in model.K for (i,j) in model.A) - sum(model.w[k,t] + sum(model.g_plus[k,s,t] - model.g_minus[k,s,t] for s in model.S) - sum(model.g_out[k,o,t] for o in model.O) for k in model.K) for t in model.T)
    # Objetive 2
    def objective2(model):
        return sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T)
    # Objective 3
    def objective3(model):
        return 1*(sum(sum(model.alpha * model.tau[i,j] * model.x[k,i,j,t] for k in model.K for (i,j) in model.A) - sum(model.w[k,t] + sum(model.g_plus[k,s,t] - model.g_minus[k,s,t] for s in model.S) - sum(model.g_out[k,o,t] for o in model.O) for k in model.K) for t in model.T)) - 1*(sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T))
    #model.obj = Objective(rule=objective1, sense=maximize)
    #model.obj = Objective(rule=objective2, sense=minimize)
    model.obj = Objective(rule=objective3, sense=maximize)

    # Constraints
    model.constraints = ConstraintList() # Create a set of constraints

    # Equation 7 (adjusted)
    for i in model.N:
        for j in model.N:
            for t in range(1, num_timesteps):  # Ensure correct range
                model.constraints.add(model.d[i,j,t+1] == model.d[i,j,t] + model.P[i,j,t] - sum(model.x[k,i,j,t] for k in model.K))

    # Equation 8
    for k in model.K:
        for i in model.N:
            for t in model.T:
                model.constraints.add(model.z[k,i,t] == 0)

    # Equation 9
    for k in model.K:
        for i in model.N:
            for t in model.T:
                model.constraints.add(model.p[k,i,t] == sum(model.x[k,i,j,t-model.tau[i,j]]+model.y[k,i,j,t-model.tau[i,j]] for (i,j) in model.A if model.tau[i,j]< t))

    # Equation 10
    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.z[k,i,t] + sum(model.x[k,j,i,t] + model.y[k,j,i,t] for i,j in model.A) for i in model.N) <= 1)

    # Equation 11
    for k in model.K:
        for i in model.N:
            for t in model.T:
                model.constraints.add(model.z[k,i,t] + model.p[k,i,t] <= 1)

    # Equation 12
    for k in model.K:
        for i in model.N:
            for t in range(1,num_timesteps):
                model.constraints.add(model.z[k,i,t+1] == model.z[k,i,t]+model.p[k,i,t] - sum(model.x[k,j,i,t] + model.y[k,j,i,t] for i,j in model.A))

    # Equation 13
    for k in model.K:
        for t in range(1,num_timesteps):
            model.constraints.add(model.eng[k,t+1] == model.eng[k,t] + sum(model.eta * model.g_plus[k,s,t] for s in model.S) - sum((1/model.eta) * model.g_minus[k,s,t] for s in model.S) - sum((1/model.eta) * model.g_out[k,o,t] for o in model.O) - (model.xi*(1 - sum(model.z[k,i,t] for i in model.N))))

    # Equation 14
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.eng[k,t] <= model.gamma_max * model.B[k])
            model.constraints.add(model.eng[k,t] >= model.gamma_min * model.B[k])

    # Equation 15
    for k in model.K:
        model.constraints.add(model.eng[k,1] == model.gamma_0[k] * model.B[k])

    # Equation 16
    for k in model.K:
        for s in model.S:
            for t in model.T:
                model.constraints.add(model.g_plus[k,s,t] <= model.rho_plus * model.u_plus[k,s,t])

    # Equation 17
    for k in model.K:
        for s in model.S:
            for t in model.T:
                model.constraints.add(model.g_minus[k,s,t] <= model.rho_minus * model.u_minus[k,s,t])

    # Equation 18
    for k in model.K:
        for o in model.O:
            for t in model.T:
                model.constraints.add(model.g_out[k,o,t] <= model.rho_out * model.u_out[k,o,t])

    # Equation 19
    for k in model.K:
        for i in model.N:
            for s in model.S:
                for o in model.O:
                    for t in range(1,num_timesteps):
                        model.constraints.add(model.u_plus[k,s,t] + model.u_minus[k,s,t] + model.u_out[k,o,t] <= model.z[k,i,t+1])

    # Equation 20
    for o in model.O:
        for t in model.T:
            model.constraints.add(sum(model.g_out[k,o,t] for k in model.K) <= model.E_req[o,t])

    # Equation 21
    for k in model.K:
        model.constraints.add(model.a[k] == model.N_Cy * model.lambda_DoD * model.B[k])

    # Equation 22
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.w[k,t]==(model.R_bat/(model.N_Cy * model.lambda_DoD)) * (sum(model.g_minus[k,s,t] for s in model.S) + sum(model.g_out[k,o,t] for o in model.O)))

    # e-constraint method
    #model.constraints.add(sum(sum(model.alpha * model.tau[i,j] * model.x[k,i,j,t] for k in model.K for (i,j) in model.A) - sum(model.w[k,t] + sum(model.g_plus[k,s,t] - model.g_minus[k,s,t] for s in model.S) - sum(model.g_out[k,o,t] for o in model.O) for k in model.K) for t in model.T) <= epsilon)
    #model.constraints.add(sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T)<=epsilon)

    opt = SolverFactory('gurobi')
    opt.options['timelimit'] = 3600
    opt.options['mipgap'] = 0.01
    opt.solve(model,tee=True,report_timing=True)
    print(model.obj())
    return model

def plot_model_variables(model, num_vehicles):
    # Extracting data for plotting
    timesteps = [t for t in model.T]
    num_vehicles = len([k for k in model.K])

    # Plotting the state of charge (SoC) for each vehicle over time
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in model.K:
        soc = [model.eng[k, t].value for t in model.T]
        ax.plot(timesteps, soc, label=f'Vehicle {k}')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('State of Charge (kWh)')
    ax.set_title('State of Charge for each Vehicle over Time')
    ax.legend()
    ax.grid(True)

    # Plotting the number of passengers waiting at each node over time
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in model.N:
        for j in model.N:
            if i != j:
                passengers = [model.d[i, j, t].value for t in model.T]
                ax.plot(timesteps, passengers, label=f'Node {i} to Node {j}')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Passengers')
    ax.set_title('Number of Passengers Waiting at each Node over Time')
    ax.legend()
    ax.grid(True)

    # Plotting the energy charged and discharged at parking spots
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in model.K:
        energy_charged = [sum(model.g_plus[k, s, t].value for s in model.S) for t in model.T]
        energy_discharged = [sum(model.g_minus[k, s, t].value for s in model.S) for t in model.T]
        ax.plot(timesteps, energy_charged, label=f'Charged by Vehicle {k}', linestyle='--')
        ax.plot(timesteps, energy_discharged, label=f'Discharged by Vehicle {k}', linestyle=':')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy Charged and Discharged at Parking Spots')
    ax.legend()
    ax.grid(True)

    # Plotting the energy offered at outage locations
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in model.K:
        energy_out = [sum(model.g_out[k, o, t].value for o in model.O) for t in model.T]
        ax.plot(timesteps, energy_out, label=f'Offered by Vehicle {k}')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy Offered (kWh)')
    ax.set_title('Energy Offered at Outage Locations')
    ax.legend()
    ax.grid(True)
    plt.show()