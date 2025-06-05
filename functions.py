import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def create_coordinates(data='Data/CT_Centroids_with_YX.csv', plot=True, num_entries=None):
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

def shortest_path(file_path='Data/Shortest_Path_Matrix.csv', num_entries=None):
    # Read the CSV file into a DataFrame, specifying the separator as ';' and setting data types for index and headers
    distances = pd.read_csv(file_path, sep=';', index_col=0, dtype={'origins': str}) / 1000
    
    # Limit the number of entries if specified
    if num_entries:
        distances = distances.head(num_entries)
        distances = distances.loc[:, distances.columns[:num_entries]]
    
    # Convert DataFrame to NumPy array
    distances_array = distances.to_numpy()
    
    return distances_array

def passenger_demand(file_path_shortest_paths='Data/Shortest_Path_Matrix.csv', hours_experiment=24, plot=True, num_entries=None):
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
        file_path_demand = f'Data/24_1hour_od/car_{hour}h.csv'

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
    data = pd.read_excel('Data/inputs.xlsx')
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
    
    return data #, battery, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD

def solveModel(epsilon, battery, passengers, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD, num_nodes, num_vehicles, num_timesteps):
    model = pyo.ConcreteModel()
    print('Solving with ɛ =', epsilon)
    
    # Sets
    model.N = pyo.RangeSet(num_nodes) # set of nodes
    model.S = pyo.RangeSet(num_nodes) # subset of parking spots
    model.O = pyo.RangeSet(num_nodes) # subset of outage locations
    model.S_O = pyo.RangeSet(num_nodes) # intersection of parking spots and outage
    model.A = pyo.Set(dimen=2, initialize=lambda model: [(i, j) for i in model.N for j in model.N if i != j]) # set of arcs
    model.K = pyo.RangeSet(num_vehicles) # set of vehicles
    model.T = pyo.RangeSet(num_timesteps) # set of timesteps

    # Parameters
    model.B = pyo.Param(model.K, initialize=lambda model, k: battery[k-1])  # Battery capacity of vehicle k [kWh]
    model.P = pyo.Param(model.N, model.N, model.T, initialize=lambda model, i, j, t: passengers[i-1, j-1, t-1])  # Number of passengers arriving at node i with destination j at time t
    model.pi_plus = pyo.Param(model.T, initialize=lambda model, t: pi_plus[t-1])  # Price to buy energy (grid) at time t [CA$/kWh]
    model.pi_minus = pyo.Param(model.T, initialize=lambda model, t: pi_minus[t-1])  # Price to sell energy (grid) at time t [CA$/kWh]
    model.pi_out = pyo.Param(model.T, initialize=lambda model, t: pi_out[t-1])  # Price to sell energy (outage) at time t [CA$/kWh]
    model.alpha = pyo.Param(initialize=alpha)  # Price paid by each customer per trip [CA$/time]
    model.E_req = pyo.Param(model.N, model.T, initialize=lambda model, n, t: E_req[n-1, t-1])  # Energy required in an affected zone o per timestep t [kWh]
    model.gamma_max = pyo.Param(initialize=gamma_max)  # Maximum state-of-charge [%]
    model.gamma_min = pyo.Param(initialize=gamma_min)  # Minimum state-of-charge [%]
    model.gamma_0 = pyo.Param(model.K, initialize=lambda model, k: gamma_0[k-1])  # Initial state-of-charge for each vehicle k [%]
    model.tau = pyo.Param(model.N, model.N, initialize=lambda model, i, j: tau[i-1, j-1])  # Travel time between node i to node j
    model.xi = pyo.Param(initialize=xi)  # Energy consumption rate [kWh/km]
    model.rho_plus = pyo.Param(initialize=rho_plus)  # Charging rate (parking spots) [kW]
    model.rho_minus = pyo.Param(initialize=rho_minus)  # Discharging rate (parking spots) [kW]
    model.rho_out = pyo.Param(initialize=rho_out)  # Discharging rate (outage location) [kW]
    model.eta = pyo.Param(initialize=eta)  # Charger efficiency [%]
    model.R_bat = pyo.Param(initialize=R_bat)  # Battery replacement costs [CA$/kWh]
    model.N_Cy = pyo.Param(initialize=N_Cy)  # Number of cycles until end-of-life
    model.lambda_DoD = pyo.Param(initialize=lambda_DoD)  # Depth of discharge [%]

    # Variables
    model.x = pyo.Var(model.K, model.N, model.N, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is carrying passengers from node i to node j at timestep t {0,1}
    model.y = pyo.Var(model.K, model.N, model.N, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is reallocating (empty) from node i to node j at timestep t {0,1}
    model.z = pyo.Var(model.K, model.N, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k in parked at node i at timestep t {0,1}
    model.p = pyo.Var(model.K, model.N, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is arriving at node i at timestep t {0,1}
    model.u_plus = pyo.Var(model.K, model.S, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is charging at node s at timestep t {0,1}
    model.u_minus = pyo.Var(model.K, model.S, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is discharging at node s at timestep t {0,1}
    model.u_out = pyo.Var(model.K, model.O, model.T, within=pyo.Binary)  # Binary variable indicating if vehicle k is offering energy in affected zone o at time t {0,1}
    model.d = pyo.Var(model.N, model.N, model.T, within=pyo.NonNegativeIntegers)  # Number of passengers waiting at node i with destination j at timestep t
    model.eng = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)  # Energy of vehicle k at timestep t [kWh]
    model.g_plus = pyo.Var(model.K, model.S, model.T, within=pyo.NonNegativeReals)  # Energy charged from vehicle k at node s at timestep t [kWh]
    model.g_minus = pyo.Var(model.K, model.S, model.T, within=pyo.NonNegativeReals)  # Energy discharged from vehicle k at node s at timestep t [kWh]
    model.g_out = pyo.Var(model.K, model.O, model.T, within=pyo.NonNegativeReals)  # Total energy offered from vehicle k to affected zone o at timestep t [kWh]
    model.a = pyo.Var(model.K, within=pyo.NonNegativeReals) # Total energy taken from a vehicle k battery through its lifespan [kWh]
    model.w = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)  # Battery degradation costs of vehicle k in timestep t to offer energy to the grid [CA$]

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
    model.obj = pyo.Objective(rule=objective3, sense=pyo.maximize)

    # Constraints
    model.constraints = pyo.ConstraintList() # Create a set of constraints

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
    model.constraints.add(sum(sum(model.alpha * model.tau[i,j] * model.x[k,i,j,t] for k in model.K for (i,j) in model.A) - sum(model.w[k,t] + sum(model.g_plus[k,s,t] - model.g_minus[k,s,t] for s in model.S) - sum(model.g_out[k,o,t] for o in model.O) for k in model.K) for t in model.T) <= epsilon)
    model.constraints.add(sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T)<=epsilon)

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
    
def solve_opt(data, pass_min=0, pass_max=5990, mult_1=1, mult_2=1, tee=False): #data, pass_min=0, pass_max=5990, mult_1=1, mult_2=1, tee=False
    model = pyo.ConcreteModel()  # create model
    # vary mult 1 to create pareto front!!!
    t = 96
    k = 100
    D_passenger = (data['Total demand'] * mult_1).tolist()
    O_energy = (data['Outage'] * mult_2).tolist()
    Riding_price = 0.5
    P_buy = 0.1
    P_sell = 0.6
    alpha = 10
    beta = 7.2
    ch_eff = 0.90
    dch_eff = 1/0.9
    gama = 1.5  # 30km/h
    E_0 = 0.2
    E_min = 0.2
    E_max = 1
    R_bat = 150
    C_bat = 80
    Cycle = 3000
    DoD = 60
    T = 96
    delta_t = 0.25

    # sets
    model.T = pyo.RangeSet(t)  # set of timesteps
    model.K = pyo.RangeSet(k)  # set of SAEVs

    # parameters
    model.D_passenger = pyo.Param(model.T, initialize=lambda model, t: D_passenger[t-1])
    model.O_energy = pyo.Param(model.T, initialize=lambda model, t: O_energy[t-1])
    
    # binary variables
    model.b = pyo.Var(model.K, model.T, within=pyo.Binary)  # serving trip indicator
    model.x = pyo.Var(model.K, model.T, domain=pyo.Binary)    # charging indicator
    model.y = pyo.Var(model.K, model.T, domain=pyo.Binary)    # discharging indicator

    # non-negative variables
    model.e = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)  # energy level of bus k at time t
    model.w_buy = pyo.Var(model.T, within=pyo.NonNegativeReals)  # electricity purchased from the grid at time t
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)  # electricity sold to the grid at time t
    model.w_riding = pyo.Var(model.T, within=pyo.NonNegativeReals)  # number of trips performed at time t
    model.d = pyo.Var(model.T, within=pyo.NonNegativeReals)  # total degradation cost of the bus k battery at time t

    # constraints
    model.constraints = pyo.ConstraintList()  # Create a set of constraints

    # Each vehicle can perform at most one action at each timestep
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.b[k, t] + model.x[k, t] + model.y[k, t] <= 1)
            
    # NEW CONSTRAINT: In the first time step, all vehicles should be charging.
    for k in model.K:
        model.constraints.add(model.x[k, 1] == 1)
        model.constraints.add(model.b[k, 1] == 0)
        model.constraints.add(model.y[k, 1] == 0)

    # Constraint for passenger demand (for each time step)
    for t in model.T:
        model.constraints.add(sum(model.b[k, t] for k in model.K) <= model.D_passenger[t])
    
    # The number of trips performed equals the number of vehicles serving at time t
    for t in model.T:
        model.constraints.add(sum(model.b[k, t] for k in model.K) == model.w_riding[t])
    
    # Passenger bounds
    model.constraints.add(sum(model.w_riding[t] for t in model.T) >= pass_min)
    model.constraints.add(sum(model.w_riding[t] for t in model.T) <= pass_max)

    # Energy update constraint for t>=2
    for k in model.K:
        for t in range(2, T + 1):
            model.constraints.add(model.e[k, t] == model.e[k, t-1] + delta_t * ch_eff * alpha * model.x[k, t] 
                                  - gama * model.b[k, t] - delta_t * dch_eff * beta * model.y[k, t])

    # Grid power balance for charging
    for t in model.T:
        model.constraints.add(delta_t * sum(ch_eff * alpha * model.x[k, t] for k in model.K) == model.w_buy[t])
    
    # Grid power balance for discharging
    for t in model.T:
        model.constraints.add(delta_t * sum(dch_eff * beta * model.y[k, t] for k in model.K) == model.w_sell[t])
    
    # Outage power limit
    for t in model.T:
        model.constraints.add(delta_t * sum(dch_eff * beta * model.y[k, t] for k in model.K) <= delta_t * model.O_energy[t])
    
    # Battery energy limits
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k, t] >= C_bat * E_min)
            model.constraints.add(model.e[k, t] <= C_bat * E_max)
    
    # Initial energy level and degradation cost
    for k in model.K:
        model.constraints.add(model.e[k, 1] == E_0 * C_bat)
    for t in model.T:
        model.constraints.add(model.d[t] == (R_bat / (Cycle * DoD)) * model.w_sell[t])

    # objective function
    def rule_obj(mod):
        return (sum(Riding_price * mod.w_riding[t] for t in mod.T) 
                - sum(P_buy * mod.w_buy[t] for t in mod.T) 
                + sum(P_sell * mod.w_sell[t] for t in mod.T)
                - sum(mod.d[t] for t in mod.T))
    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.maximize)
    
    # SOLVER
    opt = pyo.SolverFactory('gurobi')
    opt.options['timelimit'] = 600
    opt.options['mipgap'] = 0.01
    results = opt.solve(model, tee=tee)
    print(results)
    return model

def plot_variables(model, D_passenger, O_energy, t=96, k=100):

    # Minimalist style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'none',
        'axes.grid': False,  # No grid
        'font.size': 13,
        'font.family': 'DejaVu Sans',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'axes.titlepad': 10,
    })

    # Soft pastel palette
    SERVING_COLOR       = "#D37919"
    CHARGING_COLOR      = "#21C875"
    DISCHARGING_COLOR   = "#4198E4"
    IDLE_COLOR          = "#2C2C2C"
    TOTAL_ENERGY_COLOR  = "#3C83E8"
    W_BUY_COLOR         = '#8CD17D'
    W_SELL_COLOR        = '#B6992D'
    PASSENGER_COLOR     = '#499894'
    OUTAGE_COLOR        = '#E15759'

    # Time axis
    time_steps = range(1, t + 1)
    hours = [step / 4 for step in time_steps]
    bar_width = 0.18

    # Model data
    serving = [sum(pyo.value(model.b[k, t]) for k in model.K) for t in model.T]
    charging = [sum(pyo.value(model.x[k, t]) for k in model.K) for t in model.T]
    discharging = [sum(pyo.value(model.y[k, t]) for k in model.K) for t in model.T]
    idle = [k - s - c - d for s, c, d in zip(serving, charging, discharging)]
    total_energy = [sum(pyo.value(model.e[k, t]) for k in model.K) for t in model.T]
    w_buy = [pyo.value(model.w_buy[t]) for t in model.T]
    w_sell = [pyo.value(model.w_sell[t]) for t in model.T]
    serving_percentage = [(sum(pyo.value(model.b[k, t]) for k in model.K) / k) * 100 for t in model.T]
    discharging_percentage = [(sum(pyo.value(model.y[k, t]) for k in model.K) / k) * 100 for t in model.T]

    # Helper for minimalist axes with thin border
    def minimalist(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(0.7)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.tick_params(axis='both', length=0)
        ax.set_facecolor('white')

    # 1. Fleet Status Over Time
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.bar(hours, serving, width=bar_width, color=SERVING_COLOR, label='Serving', alpha=0.85)
    ax.bar(hours, charging, width=bar_width, bottom=serving, color=CHARGING_COLOR, label='Charging', alpha=0.85)
    ax.bar(hours, discharging, width=bar_width, bottom=[s + c for s, c in zip(serving, charging)],
           color=DISCHARGING_COLOR, label='Discharging', alpha=0.85)
    ax.bar(hours, idle, width=bar_width, bottom=[s + c + d for s, c, d in zip(serving, charging, discharging)],
           color=IDLE_COLOR, label='Idle', alpha=0.7)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Vehicles')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    minimalist(ax)
    legend = ax.legend(
        ncol=4,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

    # 2. Total Energy Level of the Fleet
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.plot(hours, total_energy, color=TOTAL_ENERGY_COLOR, label='Total Energy', alpha=0.9)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Energy [kWh]')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    minimalist(ax)
    legend = ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

    # 3. Charging and Discharging Power Over Time
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    ax.plot(hours, w_buy, color=W_BUY_COLOR, label='Charging (G2V)', alpha=0.8)
    ax.plot(hours, w_sell, color=W_SELL_COLOR, label='Discharging (V2G)', alpha=0.8)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Power [kW]')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    minimalist(ax)
    legend = ax.legend(
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

    # 4. Vehicles Serving (%) vs Passenger Demand
    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=300)
    ax1.bar(hours, serving_percentage, width=bar_width, color=SERVING_COLOR, label='Serving (%)', alpha=0.8)
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Vehicles [%]')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 1))
    minimalist(ax1)
    ax2 = ax1.twinx()
    ax2.plot(hours, D_passenger, color=PASSENGER_COLOR, linestyle='-', label='Passenger Demand', alpha=0.7)
    ax2.set_ylabel('Passengers')
    minimalist(ax2)
    ax2.spines['left'].set_visible(False)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

    # 5. Vehicles Discharging (%) vs Outage Demand
    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=300)
    ax1.bar(hours, discharging_percentage, width=bar_width, color=DISCHARGING_COLOR, label='Discharging (%)', alpha=0.8)
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Vehicles [%]')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 1))
    minimalist(ax1)
    ax2 = ax1.twinx()
    ax2.plot(hours, O_energy, color=OUTAGE_COLOR, linestyle='-', label='Outage Demand', alpha=0.7)
    ax2.set_ylabel('Power [kW]')
    minimalist(ax2)
    ax2.spines['left'].set_visible(False)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

    # 6. Vehicles Serving vs Power Delivered to the Grid
    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=300)
    ax1.bar(hours, serving, width=bar_width, color=SERVING_COLOR, label='Serving', alpha=0.8)
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Vehicles')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 1))
    minimalist(ax1)
    ax2 = ax1.twinx()
    ax2.plot(hours, w_sell, color=W_SELL_COLOR, linestyle='-', label='Power Delivered (V2G)', alpha=0.7)
    ax2.set_ylabel('Power [kW]')
    minimalist(ax2)
    ax2.spines['left'].set_visible(False)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        framealpha=1,
        edgecolor="#000000"
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5)
    plt.show()

def save_to_excel(model, filename='output.xlsx'):
	# Extract variables of interest
	T = list(model.T)
	K = list(model.K)
	# Serving, charging, discharging, energy, w_buy, w_sell
	serving = [sum(pyo.value(model.b[k, t]) for k in K) for t in T]
	charging = [sum(pyo.value(model.x[k, t]) for k in K) for t in T]
	discharging = [sum(pyo.value(model.y[k, t]) for k in K) for t in T]
	total_energy = [sum(pyo.value(model.e[k, t]) for k in K) for t in T]
	w_buy = [pyo.value(model.w_buy[t]) for t in T]
	w_sell = [pyo.value(model.w_sell[t]) for t in T]

	df = pd.DataFrame({
		'TimeStep': T,
		'Serving': serving,
		'Charging': charging,
		'Discharging': discharging,
		'TotalEnergy': total_energy,
		'W_buy': w_buy,
		'W_sell': w_sell
	})
	df.to_excel(filename, index=False)
	print(f"Results saved to {filename}")