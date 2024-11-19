import yaml
import torch
import numpy as np
import math

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
    
    # to perform a key-wise sum of two dicts
    def _keywise_sum(self, dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

    def rollout(self, mh_actor: MultiHeadActor, active_agents_indices: list, alg="ddpg", noisy_action=False, noise_std=0.4):
        """
        Perform a rollout of a given multiheaded actor in the MORoverEnv domain.

        Parameters:
        - mh_actor (MultiHeadActor)
        - active_agents_indices (list): List of indices that specify which agents/heads in the MHA are active and will be a part of the rollout.
        - noisy_action (bool): Whether to add noise to the actions for exploration.
        - noise_std (float): Standard deviation of the noise added for exploration.

        Returns:
        - rollout_trajectory (dict): Complete trajectory of the rollout with position, local reward, and action data of each agent.
        - global_reward (list): Reward vector that evaluates this MHA on each system-level objective.
        """

        ep_length = self.rover_env.get_ep_length()
        agent_locations = self.config['Agents']['starting_locs']  # set each agent to the starting location
        num_sensors = self.config['Agents']['num_sensors']
        observation_radii = self.config['Agents']['observation_radii']
        max_step_sizes = self.config['Agents']['max_step_sizes']
        
        cumulative_global_reward = {}  # Initialize cumulative global reward

        rollout_trajectory = {agent_idx : [] for agent_idx in active_agents_indices} # Initialise the episode's trajectory as a dict (of single agent trajectories)

        self.rover_env.reset() # reset the rover env

        for t in range(ep_length):
            observations_list = self.rover_env.generate_observations(agent_locations, num_sensors, observation_radii, normalise=True) # get each agent's observation at the current position
            
            agent_moves = []
            transitions = {} # transition = {'state' : [], 'action' : [], 'local_reward' : 0, 'next_state' : [], 'done' : False}

            for i, agent_idx in enumerate(active_agents_indices):
                # Extract the current observation for this agent
                agent_observation = torch.FloatTensor(observations_list[i]).unsqueeze(0)  # Add batch dimension for the model

                # Get the deterministic action from the actor
                action_tensor = mh_actor.clean_action(agent_observation, head=agent_idx)
                
                # If noisy_action is True, add noise to the deterministic action for exploration
                if noisy_action:
                    noise = torch.normal(mean=0.0, std=noise_std, size=action_tensor.size())
                    action_tensor = action_tensor + noise
                
                # Ensure actions are clipped to [-1, 1] after adding noise
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)

                action = action_tensor.squeeze(0).detach().numpy() # Convert action tensor to a numpy array without tracking gradient

                # Scale the action to comply with the agent's max step size
                norm = np.linalg.norm(action) # get the magnitude of the calculated move
                max_step = max_step_sizes[i]
                scaling_factor = (max_step / norm) if norm > 0 else 0 # the factor by which the moves should be scaled
                scaled_action = action * scaling_factor # multiply each member of the action by the scaling factor

                state = np.array(observations_list[i], dtype=np.float32)
                state = torch.tensor(state, dtype=torch.float32)

                action = np.array(action, dtype=np.float32)
                action = torch.tensor(action, dtype=torch.float32)

                # Construct the transition dictionary for the current agent
                transitions[agent_idx] = {
                    'state': state,
                    'action': action,
                    'local_reward' : None, # Will be applied later
                    'next_state': [],
                    'done': False
                }

                # Add scaled action to the list of agent moves
                agent_moves.append(scaled_action)
  
            agent_locations = self.rover_env.update_agent_locations(agent_locations, agent_moves, max_step_sizes) # get updated agent locations based on moves

            local_rewards = self.rover_env.get_local_rewards(agent_locations) # get local rewards for this location configuration of agents
            
            done = (t == ep_length - 1) # is the episode complete?

            # Get the global reward and update the cumulative global reward
            global_reward = self.rover_env.get_global_rewards(rov_locations=agent_locations, timestep=t)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)

            # Prepare for next state's observations (after environment update)
            next_observations_list = self.rover_env.generate_observations(agent_locations, num_sensors, observation_radii, normalise=True)

            # Update each agent's transition dictionary with next state and done
            for i, agent_idx in enumerate(active_agents_indices):
                next_state = np.array(next_observations_list[i], dtype=np.float32) if next_observations_list else np.array([], dtype=np.float32)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                transitions[agent_idx]['local_reward'] = local_rewards[i]
                transitions[agent_idx]['next_state'] =  next_state
                transitions[agent_idx]['done'] = done

                # Append the transition to the agent's trajectory
                rollout_trajectory[agent_idx].append(transitions[agent_idx])

        return rollout_trajectory, cumulative_global_reward
