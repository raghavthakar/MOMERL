import yaml
import copy
import math
import numpy as np

class POI:
    def __init__(self, obj, location, radius, coupling, obs_window, reward, repeat):
        """
        Parameters:
        - obj (int): Objective for this POI (must be > 0).
        - location (list): Coordinates for the POI.
        - radius (float or int): Radius around location within which the POI can be observed (non-negative).
        - coupling (int): Number of simultaneous observations required (non-negative).
        - obs_window (list): Time window for observation.
        - reward (float or int): Reward for successful observation (non-negative).
        - repeat (bool): Whether the POI can be observed more than once in an episode.
        """
        if not (isinstance(obj, (int, np.int16, np.int32, np.int64)) and obj > 0):
            raise ValueError('Objective for POI must be an integer > 0.')
        if not (isinstance(location, list)):
            raise ValueError('Location must be a list of positive numbers.')
        if not ((isinstance(radius, (float, int, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)) and radius >= 0)):
            raise ValueError('Radius must be a non-negative number.')
        if not (isinstance(coupling, (int, np.int16, np.int32, np.int64)) and coupling >= 0):
            raise ValueError('Coupling must be a non-negative integer.')
        if not isinstance(obs_window, list):
            raise ValueError('Observation window must be a list.')
        if not ((isinstance(reward, (float, int, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)) and reward >= 0)):
            raise ValueError('Reward must be a non-negative number.')
        if not isinstance(repeat, bool):
            raise ValueError('Repeat must be a boolean.')

        self.obj = obj
        self.location = location
        self.radius = radius
        self.coupling = coupling
        self.obs_window = obs_window
        self.reward = reward
        self.repeat = repeat

        # Save initial state for resetting
        self._initial_obs_window = copy.deepcopy(obs_window)

    def get_reward(self, rov_locations, timestep):
        """
        Get the reward value from the POI for a configuration of rover locations and point in the episode.

        Parameters:
        - rov_locations (list): Positions of each rover in the environment. Each position is a list of integers > 0.
        - timestep (int): Current timestep in the environment.

        Returns:
        - reward (float): The reward value based on the observation conditions.
        """
        # Type and value checks
        if not isinstance(rov_locations, list):
            raise ValueError("rov_locations must be a list of rover positions.")
    
        for loc in rov_locations:
            if not isinstance(loc, list) or len(loc) != len(self.location):
                raise ValueError("Each rover location must be a list matching the POI's dimensionality.")
            if not all(isinstance(coord, (int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)) and coord > 0 for coord in loc):
                print(loc)
                raise ValueError("All coordinates in each location must be numbers greater than zero.")
        
        # Check that timestep is a non-negative integer
        if not isinstance(timestep, (int, np.int16, np.int32, np.int64)) or timestep < 0:
            raise ValueError("timestep must be a non-negative integer.")

        # Check if the current timestep is within the observation window
        if timestep < self.obs_window[0] or timestep > self.obs_window[1]:
            return 0  # No reward if not within the observation window

        # Count rovers within the POI's observation radius
        observing_rovers = 0
        for loc in rov_locations:
            # Calculate Euclidean distance in N dimensions
            distance = sum((loc_i - poi_i) ** 2 for loc_i, poi_i in zip(loc, self.location)) ** 0.5
            if distance <= self.radius:
                observing_rovers += 1

        # Check if enough rovers are observing the POI (coupling condition)
        if observing_rovers >= self.coupling:
            # If the POI can only be observed once, disable reward for subsequent observations
            if not self.repeat:
                self.obs_window = [float('inf'), float('inf')]  # Effectively disables further rewards
            return self.reward  # Return the reward since conditions are met

        return 0  # No reward if conditions are not met

    def reset(self):
        """Reset the POI to its initial state."""
        self.obs_window = copy.deepcopy(self._initial_obs_window)


class MORoverEnv:
    def __init__(self, config_filename):
        if not isinstance(config_filename, str):
            raise ValueError('Rover configuration filename must be a string.')

        self.config_filename = config_filename
        self._load_config()  # Initial loading of the environment configuration

    def _load_config(self):
        """Load environment configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            config_data = yaml.safe_load(config_file)
            print('[MORoverEnv]: YAML config read.')

        # Initialize environment properties
        self.num_objs = config_data['Meta']['num_objs']
        self.dimensions = config_data['Environment']['dimensions']
        self.ep_length = config_data['Environment']['ep_length']
        self.timestep_penalty = config_data['Environment']['timestep_penalty']

        # Initialize POIs and store initial configuration
        self.pois = [POI(**poi) for poi in config_data['Environment']['pois']]
        self._initial_pois = copy.deepcopy(self.pois)  # Save initial state for reset

    def reset(self):
        """Reset the environment to its initial configuration."""
        # Reload the environment configuration
        self._load_config()
        # Reset each POI to its original state
        for poi in self.pois:
            poi.reset()
    
    def get_global_rewards(self, rov_locations, timestep):
        """
        Calculate and return the net reward vector for a list of rover positions at a given timestep.

        Parameters:
        - rov_locations (list): List of rover positions, where each position is a list of floats > 0.
        - timestep (int): Current timestep in the environment.

        Returns:
        - reward_vector (dict): Dictionary where keys are objectives (obj) and values are cumulative rewards for each objective.
        """
        # Initialize the reward vector as a dictionary with objectives as keys and zeroed cumulative rewards
        reward_vector = {poi.obj: 0 for poi in self.pois}

        # Calculate rewards for each POI and update the reward vector
        for poi in self.pois:
            reward = poi.get_reward(rov_locations, timestep)
            if reward > 0:
                reward_vector[poi.obj] += reward

        return reward_vector
    
    def get_local_rewards(self, rov_locations):
        """
        Calculate and return the inverse of distance to closest POI for each rover in a list of rover positions.
        
        Parameters:
        - rov_locations (list): List of rover positions, where each position is a list of floats > 0.
        
        Returns:
        - local_rewards (list): List of local rewards where each element corresponds to a rover.
        """
        # Type and value checks
        if not isinstance(rov_locations, list):
            raise ValueError("rov_locations must be a list of rover positions.")
        num_dimensions = len(self.dimensions)
        for idx, loc in enumerate(rov_locations):
            if not (isinstance(loc, list) and len(loc) == num_dimensions):
                raise ValueError(f"Rover position at index {idx} must be a list of length {num_dimensions}.")
            if not all(isinstance(coord, (int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)) and coord > 0 for coord in loc):
                raise ValueError(f"All coordinates for rover at index {idx} must be numbers greater than zero.")

        # Initialize an empty local reward list
        local_rewards = []

        # Calculate local reward for each rover
        for rover_pos in rov_locations:
            min_distance = float('inf')
            for poi in self.pois:
                # Calculate Euclidean distance in N dimensions
                distance = math.sqrt(sum((r - p) ** 2 for r, p in zip(rover_pos, poi.location)))
                if distance < min_distance:
                    min_distance = distance

            # Inverse of distance as local reward, handle division by zero
            if min_distance > 0:
                local_reward = 1.0 / min_distance # NOTE: Reward value could be huge
            else:
                local_reward = float('inf')  # If distance is zero, assign infinite reward or a predefined max value
            local_rewards.append(local_reward)

        return local_rewards

    
    def update_agent_locations(self, agent_locations, agent_deltas, max_step_sizes):
        """
        Update agent locations based on movement deltas and max step sizes (Euclidean), respecting environment boundaries.

        Parameters:
        - agent_locations (list): Current positions of each agent, where each position is a list of coordinates.
        - agent_deltas (list): Movement deltas for each agent, with each delta list matching the dimensions of agent locations.
        - max_step_sizes (list): Maximum allowable Euclidean step size for each agent.

        Returns:
        - updated_locations (list): New list of agent locations after applying scaled deltas and respecting boundaries.
        """
        if not (isinstance(agent_locations, list) and isinstance(agent_deltas, list) and isinstance(max_step_sizes, list)):
            raise ValueError("agent_locations, agent_deltas, and max_step_sizes must all be lists.")
        if not (len(agent_locations) == len(agent_deltas) == len(max_step_sizes)):
            raise ValueError("agent_locations, agent_deltas, and max_step_sizes must have the same length.")

        updated_locations = []

        for loc, delta, max_step in zip(agent_locations, agent_deltas, max_step_sizes):
            if not (len(loc) == len(delta) == len(self.dimensions)):
                raise ValueError("Each agent's location and delta must match the environment's dimensionality.")

            # Calculate the Euclidean norm of the delta
            norm = math.sqrt(sum(d_i ** 2 for d_i in delta))
            
            # Scale delta if norm exceeds max_step
            if norm > max_step:
                scale = max_step / norm
                delta = [d_i * scale for d_i in delta]

            # Calculate new location with boundaries applied
            new_location = []
            for i, (coord, delta_i, max_dim) in enumerate(zip(loc, delta, self.dimensions)):
                updated_coord = coord + delta_i
                updated_coord = max(0, min(updated_coord, max_dim - 1))  # Enforce boundaries
                new_location.append(updated_coord)

            updated_locations.append(new_location)

        return updated_locations

    def generate_observations(self, rover_locations, num_sensors_list, observation_radius_list):
        """
        Generate observations for all rovers based on their positions, sensors, and observation radii.

        Parameters:
        - rover_locations (list): Positions of each rover. Each position is a list of coordinates.
        - num_sensors_list (list): Number of sensors (cones) for each rover.
        - observation_radius_list (list): Observation radius for each rover.

        Returns:
        - observations_list (list): List of observations for each rover.
          Each observation is a list containing the rover's position followed by counts of POIs and agents.
        """
        if not (isinstance(rover_locations, list) and isinstance(num_sensors_list, list) and isinstance(observation_radius_list, list)):
            raise ValueError("rover_locations, num_sensors_list, and observation_radius_list must all be lists.")
        if not (len(rover_locations) == len(num_sensors_list) == len(observation_radius_list)):
            raise ValueError("rover_locations, num_sensors_list, and observation_radius_list must have the same length.")

        observations_list = []
        num_dimensions = len(self.dimensions)

        for idx, (rover_pos, num_sensors, obs_radius) in enumerate(zip(rover_locations, num_sensors_list, observation_radius_list)):
            if not (isinstance(rover_pos, list) and len(rover_pos) == num_dimensions):
                raise ValueError(f"Rover position at index {idx} must be a list of length {num_dimensions}.")
            if not isinstance(num_sensors, (int, np.int16, np.int32, np.int64)) or num_sensors <= 0:
                raise ValueError(f"Number of sensors for rover at index {idx} must be a positive integer.")
            if not (isinstance(obs_radius, (int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)) and obs_radius >= 0):
                raise ValueError(f"Observation radius for rover at index {idx} must be a non-negative number.")

            observation = []

            # Add the rover's own position
            observation.extend(rover_pos)

            # Initialize observations
            if num_dimensions == 1:
                # 1D environment
                poi_count = 0
                agent_count = 0

                # Count POIs within observation radius
                for poi in self.pois:
                    distance = abs(poi.location[0] - rover_pos[0])
                    if distance <= obs_radius:
                        poi_count += 1

                # Count other agents within observation radius
                for other_idx, other_pos in enumerate(rover_locations):
                    if other_idx == idx:
                        continue  # Skip self
                    distance = abs(other_pos[0] - rover_pos[0])
                    if distance <= obs_radius:
                        agent_count += 1

                # Add counts to observations
                observation.append(poi_count)
                observation.append(agent_count)

            elif num_dimensions == 2:
                # 2D environment
                num_cones = num_sensors
                cone_angle = 360.0 / num_cones
                poi_observations = [0] * num_cones
                agent_observations = [0] * num_cones

                # Count POIs within observation radius and cones
                for poi in self.pois:
                    dx = poi.location[0] - rover_pos[0]
                    dy = poi.location[1] - rover_pos[1]
                    distance = math.hypot(dx, dy)

                    if distance <= obs_radius:
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        cone_index = int(angle // cone_angle)
                        poi_observations[cone_index] += 1

                # Count other agents within observation radius and cones
                for other_idx, other_pos in enumerate(rover_locations):
                    if other_idx == idx:
                        continue  # Skip self
                    dx = other_pos[0] - rover_pos[0]
                    dy = other_pos[1] - rover_pos[1]
                    distance = math.hypot(dx, dy)

                    if distance <= obs_radius:
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        cone_index = int(angle // cone_angle)
                        agent_observations[cone_index] += 1

                # Add counts to observations
                observation.extend(poi_observations)
                observation.extend(agent_observations)
            else:
                raise NotImplementedError("Observation generation is only implemented for 1D and 2D environments.")

            observations_list.append(observation)

        return observations_list

    def get_ep_length(self):
        """
        Get the length of the episode of this instance of the MORoverEnv domain.
        """
        return self.ep_length