# main.py

import math
from MORoverEnv import MORoverEnv

def main():
    # Initialize the environment
    env = MORoverEnv('/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml')
    
    # Number of rovers
    num_rovers = 3
    
    # Initial rover positions (assuming 2D environment)
    rover_locations = [
        [10, 10],
        [30, 40],
        [21, 61]
    ]
    
    # Movement deltas for each rover (desired movements)
    agent_deltas = [
        [5, 5],     # Rover 0 moves +5 in x and y
        [-10, 0],   # Rover 1 moves -10 in x
        [0, -20]    # Rover 2 moves -20 in y
    ]
    
    # Maximum Euclidean step sizes for each rover
    max_step_sizes = [7, 7, 7]  # Max step size of 7 units per rover
    
    # Number of sensors per rover
    num_sensors_list = [4, 6, 8]  # Rovers have 4, 6, and 8 sensors respectively
    
    # Observation radius per rover
    observation_radius_list = [20.0, 25.0, 30.0]
    
    # Timestep
    timestep = 0
    
    print("Initial Rover Positions:")
    for idx, loc in enumerate(rover_locations):
        print(f"Rover {idx}: {loc}")
    
    # Update rover positions
    updated_rover_locations = env.update_agent_locations(rover_locations, agent_deltas, max_step_sizes)
    
    print("\nUpdated Rover Positions:")
    for idx, loc in enumerate(updated_rover_locations):
        print(f"Rover {idx}: {loc}")
    
    # Generate observations for each rover
    observations = env.generate_observations(updated_rover_locations, num_sensors_list, observation_radius_list)
    
    print("\nObservations for Each Rover:")
    for idx, obs in enumerate(observations):
        print(f"Rover {idx} Observations: {obs}")
    
    # Calculate net rewards
    net_rewards = env.get_global_rewards(updated_rover_locations, timestep)
    
    print("\nNet Rewards:")
    for obj, reward in net_rewards.items():
        print(f"Objective {obj}: Reward {reward}")
    
    # Calculate local rewards for each rover
    local_rewards = env.get_local_rewards(updated_rover_locations)
    
    print("\nLocal Rewards for Each Rover:")
    for idx, reward in enumerate(local_rewards):
        print(f"Rover {idx}: Local Reward {reward}")
    
    # Increment timestep
    timestep += 1

if __name__ == "__main__":
    main()
