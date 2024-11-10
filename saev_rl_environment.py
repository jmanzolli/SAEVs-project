import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.spatial import distance

class SAEVEnv(gym.Env):
    def __init__(self):
        super(SAEVEnv, self).__init__()
        
        # Define the action and observation space
        # Actions: Assign vehicles, Charging decisions, etc.
        self.action_space = spaces.MultiDiscrete([5] * 5)  # Example: [idle, move passenger, reallocate, charge, discharge]
        
        # Observations: Vehicle locations, SOC, passenger requests, outages, etc.
        self.observation_space = spaces.Box(low=np.array([0] * 15), high=np.array([1] * 15), dtype=np.float32)
        
        # Environment state variables
        self.num_vehicles = 5
        self.num_nodes = 10
        
        # Parameters from the problem description
        self.battery_capacity = [100] * self.num_vehicles  # Battery capacity for each vehicle
        self.passenger_requests = None  # Placeholder for passenger requests, initialized in reset
        self.vehicle_positions = [0] * self.num_vehicles  # Initial positions
        self.soc = [0.5] * self.num_vehicles  # State of charge for each vehicle
        self.energy_trading_price = 2  # Price for selling energy during outages
        self.battery_degradation_cost = 1  # Cost for battery degradation per unit energy provided
        self.passenger_trip_price = 5  # Price per passenger trip
        self.min_soc = 0.2  # Minimum state of charge
        self.max_soc = 1.0  # Maximum state of charge
        self.vehicle_status = [0] * self.num_vehicles  # Status of each vehicle (idle, moving, reallocating, etc.)
        self.remaining_travel_time = [0] * self.num_vehicles  # Remaining travel time for vehicles
        
        # Distance matrix between nodes (example values)
        self.distance_matrix = np.random.randint(1, 10, size=(self.num_nodes, self.num_nodes))  # Distance between nodes
        np.fill_diagonal(self.distance_matrix, 0)  # Distance from a node to itself is 0
        
        # Charging station locations
        self.charging_stations = [0, 2, 5, 7]  # Nodes where charging stations are available
        
        # Outage-related variables
        self.outage_nodes = []  # Nodes where outages are occurring
        self.outage_durations = {}  # Dictionary to track outage duration per node
        self.max_outage_energy = 50  # Maximum energy that can be delivered during an outage

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.passenger_requests = np.zeros((self.num_nodes, self.num_nodes))  # Requests between nodes
        # Reset the state of the environment to an initial state
        self.passenger_requests.fill(0)
        self.vehicle_positions = [0] * self.num_vehicles
        self.soc = [0.5] * self.num_vehicles
        self.vehicle_status = [0] * self.num_vehicles
        self.remaining_travel_time = [0] * self.num_vehicles
        self.outage_nodes = []
        self.outage_durations = {}
        
        # Example state: Concatenate all observations into a single vector
        self.state = np.concatenate([self.vehicle_positions, self.soc, self.remaining_travel_time]).astype(np.float32).astype(np.float32).astype(np.float32).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Implement the logic for taking a step in the environment
        # Apply actions: idle, move passenger, reallocate, charge, discharge
        
        rewards = 0
        done = False
        
        # Generate new passenger requests dynamically
        self.generate_passenger_requests()
        # Update outages dynamically
        self.update_outages()
        
        for i, act in enumerate(action):
            if self.remaining_travel_time[i] > 0:
                # Vehicle is currently traveling, decrement travel time
                self.remaining_travel_time[i] -= 1
                if self.remaining_travel_time[i] == 0:
                    # Travel completed, set status to idle
                    self.vehicle_status[i] = 0
                continue

            current_node = self.vehicle_positions[i]

            if act == 0:
                # Idle - no action, no reward
                self.vehicle_status[i] = 0
            elif act == 1:
                # Move vehicle with a passenger
                destination_node = (current_node + 1) % self.num_nodes  # Example of choosing a destination node
                if self.passenger_requests[current_node, destination_node] > 0:
                    self.passenger_requests[current_node, destination_node] -= 1  # Serve one passenger
                    rewards += self.passenger_trip_price  # Reward for moving a passenger
                else:
                    rewards -= 2  # Penalty for attempting a trip without passengers

                self.vehicle_positions[i] = destination_node
                self.soc[i] = max(self.min_soc, self.soc[i] - 0.05)  # Decrease SOC due to trip
                self.vehicle_status[i] = 1
                self.remaining_travel_time[i] = self.distance_matrix[current_node, destination_node]  # Set travel time based on distance matrix
            elif act == 2:
                # Reallocate vehicle (empty move)
                destination_node = (current_node + 1) % self.num_nodes  # Example of choosing a destination node
                self.vehicle_positions[i] = destination_node
                rewards -= 1  # Penalty for empty reallocation
                self.soc[i] = max(self.min_soc, self.soc[i] - 0.03)  # Decrease SOC due to reallocation
                self.vehicle_status[i] = 2
                self.remaining_travel_time[i] = self.distance_matrix[current_node, destination_node]  # Set travel time based on distance matrix
            elif act == 3:
                # Charge vehicle
                if current_node in self.charging_stations:
                    if self.soc[i] < self.max_soc:
                        self.soc[i] = min(self.max_soc, self.soc[i] + 0.1)  # Increment SOC by 10%
                        rewards -= 0.5  # Example cost for charging
                    self.vehicle_status[i] = 3
                else:
                    # Reallocate to the nearest charging station
                    nearest_station = self.find_nearest_charging_station(current_node)
                    self.vehicle_positions[i] = nearest_station
                    self.vehicle_status[i] = 2
                    self.remaining_travel_time[i] = self.distance_matrix[current_node, nearest_station]  # Set travel time based on distance matrix
                    rewards -= 1  # Penalty for reallocation to charging station
            elif act == 4:
                # Discharge vehicle (provide energy during outage)
                if current_node in self.outage_nodes:
                    if self.soc[i] > 0.8:
                        energy_delivered = min(self.max_outage_energy, self.soc[i] - self.min_soc)  # Limit energy delivered
                        rewards += self.energy_trading_price * energy_delivered  # Positive reward for selling energy during outages
                        self.soc[i] = max(self.min_soc, self.soc[i] - energy_delivered)  # Decrease SOC due to discharging
                        rewards -= self.battery_degradation_cost * energy_delivered  # Negative reward for battery degradation
                    self.vehicle_status[i] = 4
                else:
                    # Reallocate to the nearest outage node
                    if self.outage_nodes:
                        nearest_outage = self.find_nearest_outage_node(current_node)
                        self.vehicle_positions[i] = nearest_outage
                        self.vehicle_status[i] = 2
                        self.remaining_travel_time[i] = self.distance_matrix[current_node, nearest_outage]  # Set travel time based on distance matrix
                        rewards -= 1  # Penalty for reallocation to outage node

            # Enforce SOC bounds
            if self.soc[i] < self.min_soc:
                self.soc[i] = self.min_soc  # Ensure SOC does not go below minimum

        # Penalty for waiting passengers
        waiting_passengers = np.sum(self.passenger_requests)
        rewards -= waiting_passengers * 0.1  # Penalty for passengers waiting for service

        # Update state
        self.state = np.concatenate([self.vehicle_positions, self.soc, self.remaining_travel_time])

        # Determine if episode is done
        if sum(self.soc) < self.num_vehicles * self.min_soc:  # Example condition to end episode
            done = True
            rewards -= 10  # Penalty for low energy

        terminated = done
        truncated = False
        return self.state, rewards, terminated, truncated, {}

    def generate_passenger_requests(self):
        # Generate new passenger requests dynamically at each timestep
        # For example, there is a 20% chance that a new passenger request is added between any two nodes
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.rand() < 0.2:  # 20% probability of new request
                    self.passenger_requests[i, j] += 1

    def update_outages(self):
        # Update outages dynamically at each timestep
        # Introduce new outages with a 10% probability, and decrement existing outage durations
        for node in list(self.outage_durations.keys()):
            self.outage_durations[node] -= 1
            if self.outage_durations[node] <= 0:
                del self.outage_durations[node]
                self.outage_nodes.remove(node)

        # Introduce new outages with 10% probability per node
        for node in range(self.num_nodes):
            if node not in self.outage_nodes and np.random.rand() < 0.1:
                self.outage_nodes.append(node)
                self.outage_durations[node] = np.random.randint(3, 7)  # Outage duration between 3 and 7 timesteps

    def find_nearest_charging_station(self, current_node):
        # Find the nearest charging station to the current node
        distances = [self.distance_matrix[current_node, station] for station in self.charging_stations]
        nearest_station_index = np.argmin(distances)
        return self.charging_stations[nearest_station_index]

    def find_nearest_outage_node(self, current_node):
        # Find the nearest outage node to the current node
        distances = [self.distance_matrix[current_node, outage] for outage in self.outage_nodes]
        nearest_outage_index = np.argmin(distances)
        return self.outage_nodes[nearest_outage_index]

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f"Vehicle positions: {self.vehicle_positions}")
        print(f"State of charge: {self.soc}")
        print(f"Vehicle status: {self.vehicle_status}")
        print(f"Remaining travel time: {self.remaining_travel_time}")
