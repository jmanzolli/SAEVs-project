import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

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