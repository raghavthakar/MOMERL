import yaml
import torch
import numpy as np

from multiheaded_actor import MultiHeadActor
from MORoverEnv import MORoverEnv

class MORoverInterface():
    def __init__(self, rover_config_filename):
        """
        Initialise the MOROverInterface class with its instance of the MOROverEnv Domain.
        Setup an internal reference to the rover config file
        """
        self.rover_env = MORoverEnv(rover_config_filename)
        with open(rover_config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def rollout(self, mh_actor: MultiHeadActor, active_agents_indices: list):
        """
        Perform a rollout of a given multiheaded actor in the MORoverEnv domain.

        Parameters:
        - mh_actor (MultiHeadActor)
        - active_agents_indices (list): List of indices that specify which agents/heads in the MHA are active and will be a part of the rollout.

        Returns:
        - rollout_trajectory (dict): Complete trajectory of the rollout with position, local reward, and action data of each agent.
        - global_reward (list): Reward vector that evaluates this MHA on each system-level objective.
        """

        ep_length = self.rover_env.get_ep_length()
        agent_locations = self.config['Agents']['starting_locs'] # set each agent to the starting location
        num_sensors = self.config['Agents']['num_sensors']
        observation_radii = self.config['Agents']['observation_radii']
        max_step_sizes = self.config['Agents']['max_step_sizes']
        self.rover_env.reset() # reset the rover env

        for t in range(ep_length):
            # print(agent_locations)

            observations = self.rover_env.generate_observations(agent_locations, num_sensors, observation_radii) # get each agent's observation at the current position
            
            observations_tensor = torch.tensor(observations, dtype=torch.float32) # Convert observations to a torch tensor (required for the actor model)# Convert observations to a torch tensor (required for the actor model)
            
            observation_size = len(observations) // len(active_agents_indices) # size of each agent's observation

            agent_moves = []
            for i in range(len(active_agents_indices)):
                action = mh_actor.clean_action(observations_tensor[observation_size * i : observation_size * (i + 1)], active_agents_indices[i]) # add the agent's actions to the list
                
                action = action.squeeze(0).detach().numpy() # Convert action tensor to a numpy array without tracking gradient

                norm = np.linalg.norm(action) # get the magnitude of the calculated move
                scaling_factor = max_step_sizes[i] / norm # the factor by which the moves should be scaled
                scaled_action = action * scaling_factor # multiply each member of the action by the scaling factor

                # Add scaled action to the list of agent moves
                agent_moves.append(scaled_action)
  
            agent_locations = self.rover_env.update_agent_locations(agent_locations, agent_moves, max_step_sizes)

        return None, self.rover_env.get_global_rewards(agent_locations, ep_length - 1)