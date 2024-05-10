import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from pyomo.environ import *

def generate_random_coordinates(num_nodes, min_coord=45.504157, max_coord=45.523157, min_lon=-73.638946, max_lon=-73.738946):
    coordinates = {}
    for i in range(num_nodes):
        lat = np.random.uniform(min_coord, max_coord)
        lon = np.random.uniform(min_lon, max_lon)
        coordinates[i] = (lat, lon)
    return coordinates

def create_distance_matrix(num_nodes):
    coordinates = generate_random_coordinates(num_nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0  # Distance from a node to itself is 0
            else:
                # Extract coordinates from tuples
                lat_i, lon_i = coordinates[i]
                lat_j, lon_j = coordinates[j]
                # Calculate Euclidean distance between node i and node j
                distance_matrix[i, j] = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)

    return distance_matrix, coordinates

def plot_nodes(coordinates):
    plt.figure(figsize=(8, 6))
    for i, (x, y) in coordinates.items():
        plt.scatter(x, y, color='blue')
        plt.text(x, y, f'{i}', fontsize=12, ha='center', va='bottom')
        for j, (x2, y2) in coordinates.items():
            if i != j:
                plt.plot([x, x2], [y, y2], color='gray', linestyle='-', linewidth=0.5)
    
    plt.title('Nodes and their Connections')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def create_map(coordinates):
    # Create a map centered around the mean of the coordinates
    center_lat = np.mean([lat for lat, lon in coordinates.values()])
    center_lon = np.mean([lon for lat, lon in coordinates.values()])
    map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add markers for each node
    for node, (lat, lon) in coordinates.items():
        folium.Marker([lat, lon], tooltip=f'Node {node}').add_to(map)

    # Display the map
    return map

def createData(distance_matrix, num_of_pick_ups = 6, num_parks = 3, num_out = 1, num_vehicles = 3, num_timesteps = 96):
    num_nodes = num_of_pick_ups + num_parks + num_out
    B = [40] * num_vehicles  # Battery capacity of vehicle k [kWh]
    P = np.random.choice([0, 1], size=(num_nodes, num_nodes, num_timesteps), p=[0.7,0.3])
    pi_plus = np.random.uniform(0.02, 0.05, size=num_timesteps)  # Price to buy energy (grid) at time t [CA$/kWh]
    pi_minus = np.random.uniform(0.01, 0.03, size=num_timesteps)  # Price to sell energy (grid) at time t [CA$/kWh]
    pi_out = np.random.uniform(0.015, 0.035, size=num_timesteps)  # Price to sell energy (outage) at time t [CA$/kWh]
    alpha = 0.5  # Price paid by each customer per trip [CA$/time]
    E_req = np.random.choice([0,100],size=(num_nodes, num_timesteps),p=[0.95,0.05])  # Energy required in an affected zone o per timestep t [kWh]
    gamma_max = 1  # Maximum state-of-charge [%]
    gamma_min = 0.2  # Minimum state-of-charge [%]
    gamma_0 = np.random.uniform(0.75, 1.0, size=num_vehicles)  # Initial state-of-charge for each vehicle k [%]
    tau = ((distance_matrix*100)/4).astype(int)  # Travel time between node i to node j // average speed: 12 km/h
    xi = 0.15  # Energy consumption rate [kWh/km]
    rho_plus = 10  # Charging rate (parking spots) [kW]
    rho_minus = 10  # Discharging rate (parking spots) [kW]
    rho_out = 5  # Discharging rate (outage location) [kW]
    eta = 0.9  # Charger efficiency [%]
    R_bat = 130  # Battery replacement costs [CA$/kWh]
    N_Cy = 3000  # Number of cycles until end-of-life
    lambda_DoD = 0.8 # Depth of discharge [%]
    return B, P, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD

def solveModel(epsilon, B, P, pi_plus, pi_minus, pi_out, alpha, E_req, gamma_max, gamma_min, gamma_0, tau, xi, rho_plus, rho_minus, rho_out, eta, R_bat, N_Cy, lambda_DoD, num_of_pick_ups = 6, num_parks = 3, num_out = 1, num_vehicles = 3, num_timesteps = 96):
    model = ConcreteModel()
    print('Solving with ɛ =', epsilon)
    num_nodes = num_of_pick_ups + num_parks + num_out
    model.N = RangeSet(num_nodes) # set of nodes
    model.S = Set(initialize=[i for i in model.N if i <= num_parks]) # subset of parking spots
    model.O = Set(initialize=[i for i in model.N if num_parks < i <= num_parks + num_out]) # subset of outage locations
    model.S_O = model.S | model.O # intersection of parking spots and outage
    model.Others = model.N - model.S - model.O # pick-up and drop-off locations
    model.A = Set(dimen=2, initialize=lambda model: [(i, j) for i in model.N for j in model.N if i != j]) # set of arcs
    model.A_trips = Set(dimen=2, initialize=lambda model: [(i, j) for i in model.Others for j in model.Others if i != j])
    model.K = RangeSet(num_vehicles) # set of vehicles
    model.T = RangeSet(num_timesteps) # set of timesteps

    model.B = Param(model.K, initialize=lambda model, k: B[k-1])  # Battery capacity of vehicle k [kWh]
    model.P = Param(model.N, model.N, model.T, initialize=lambda model, i, j, t: P[i-1, j-1, t-1])  # Number of passengers arriving at node i with destination j at time t
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
        return sum(sum(model.alpha * model.tau[i,j] * model.x[k,i,j,t] for k in model.K for (i,j) in model.A) - sum(model.w[k,t] + sum(model.g_plus[k,s,t] - model.g_minus[k,s,t] for s in model.S) - sum(model.g_out[k,o,t] for o in model.O) for k in model.K) for t in model.T) - sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T)
    #model.obj = Objective(rule=objective1, sense=maximize)
    model.obj = Objective(rule=objective2, sense=minimize)
    #model.obj = Objective(rule=objective3, sense=maximize)



    model.constraints = ConstraintList() # Create a set of constraints

    # Equation 7
    for i,j in model.A:
        for t in range(1,24):
            model.constraints.add(model.d[i,j,t+1] == model.d[i,j,t]+model.P[i,j,t]-sum(model.x[k,i,j,t] for k in model.K))

    # Equation 8
    for k in model.K:
        for i in model.Others:
            for t in model.T:
                model.constraints.add(model.z[k,i,t] == 0)

    # Equation 9
    for k in model.K:
        for i in model.N:
            for t in model.T:
                model.constraints.add(model.p[k,i,t] == sum(model.x[k,j,i,t-model.tau[i,j]]+model.y[k,j,i,t-model.tau[i,j]] for (j,i) in model.A if model.tau[i,j]< t))

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
            for t in range(1,24):
                model.constraints.add(model.z[k,i,t+1] == model.z[k,i,t]+model.p[k,i,t] - sum(model.x[k,j,i,t] + model.y[k,j,i,t] for i,j in model.A))

    # Equation 13
    for k in model.K:
        for t in range(1,24):
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
                    for t in range(1,24):
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
    #model.constraints.add(sum(sum(model.d[i,j,t] for (i,j) in model.A) + sum(model.tau[i,j] * model.y[k,i,j,t] for k in model.K for (i,j) in model.A) for t in model.T)<=epsilon)

    opt = SolverFactory('gurobi')
    opt.options['timelimit'] = 30
    opt.options['mipgap'] = 0.001
    results = opt.solve(model,tee=False)
    print(model.obj())
    return model