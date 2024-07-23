import numpy as np
import matplotlib.pyplot as plt
import folium
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

# Random data generation function
def generate_data(num_vehicles, num_timesteps):
    # Create an empty DataFrame to store the data
    columns = ['Vehicle ID', 'Timestep', 'SoC (%)', 'Charging Power (kW)', 'Discharging Power (kW)', 'Passengers Picked Up', 'Passengers Delivered', 'Energy Consumption (kWh)']
    data = []

    for vehicle in range(1, num_vehicles + 1):
        soc = 80  # Initial state of charge for each vehicle

        for timestep in range(1, num_timesteps + 1):
            # Randomly generate passengers picked up and delivered
            passengers_picked_up = np.random.randint(0, 5)
            passengers_delivered = np.random.randint(0, passengers_picked_up + 1)

            # Calculate energy consumption related to pick-up and delivery trips
            energy_consumption = (passengers_picked_up + passengers_delivered) * np.random.uniform(0.25, 1.5)

            # Charging only if SOC is below 25%
            if soc < 25:
                charging_power = np.random.choice([0, 5, 10])
            else:
                charging_power = 0

            # Occasional discharging (selling energy back to the grid)
            if np.random.rand() < 0.025:  # Adjust the frequency of discharging events as needed
                discharging_power = np.random.choice([0, 5, 10])
            else:
                discharging_power = 0

            # Update SOC based on energy consumption, charging, and discharging
            soc = max(0, min(100, soc - energy_consumption + charging_power - discharging_power))

            data.append([vehicle, timestep, soc, charging_power, discharging_power, passengers_picked_up, passengers_delivered, energy_consumption])
    
    return data