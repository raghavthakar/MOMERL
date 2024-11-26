import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import yaml

from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface
from multiheaded_actor import MultiHeadActor
from DDPGCritic import Critic
from ReplayBuffer import ReplayBuffer

np.random.seed(2024)
torch.manual_seed(2024)
random.seed(2024)

criterion = nn.MSELoss()

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# TODO: check to see if this soft update is correct or not
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class DDPG:
    def __init__(self, alg_config_filename, rover_config_filename, init_target_policy):
        # Initialise the test domain
        self.interface = MORoverInterface(rover_config_filename)

        # read and load the config file
        self.config_filename = alg_config_filename
        self._read_config()

        # initialize as many main critics, target critics, replya buffers, as agents on the roster
        self.main_critics = []
        self.target_critics = []
        self.main_critic_optims = []
        self.replay_buffers = []

        for i in range(self.roster_size):
            self.main_critics.append(Critic(self.interface.get_state_size(), self.interface.get_action_size(), self.critic_hidden_size))
            self.target_critics.append(Critic(self.interface.get_state_size(), self.interface.get_action_size(), self.critic_hidden_size))

            hard_update(self.target_critics[i], self.main_critics[i])

            self.main_critic_optims.append(torch.optim.Adam(self.main_critics[-1].parameters(), lr=self.critic_lr))

            self.replay_buffers.append(ReplayBuffer(10000))
        
        # one main and target policy
        # self.main_policy = MultiHeadActor(self.interface.get_state_size(), self.interface.get_action_size(), self.actor_hidden_size, self.roster_size)
        # self.target_policy = deepcopy(init_target_policy)
        self.target_policy = MultiHeadActor(self.interface.get_state_size(), self.interface.get_action_size(), self.actor_hidden_size, self.roster_size)
        hard_update(self.target_policy, init_target_policy)
        
        self.main_policy_optim = torch.optim.Adam(init_target_policy.parameters(), lr=self.actor_lr)
    
    def _read_config(self):
        """Read and load DDPG configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[DDPG]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal DDPG configuration."""
        self.actor_hidden_size = self.config_data['MHA']['hidden_size']
        self.actor_lr = self.config_data['DDPG']['actor_lr']

        self.critic_hidden_size = self.config_data['DDPG']['critic_hidden_size']
        self.soft_update_tau = self.config_data['DDPG']['soft_update_tau']
        self.critic_value_discount = self.config_data['DDPG']['critic_value_discount']
        self.critic_lr = self.config_data['DDPG']['critic_lr']
        self.tau = self.config_data['DDPG']['soft_update_tau']
        self.discount = self.config_data['DDPG']['critic_value_discount']

        self.roster_size = self.config_data['Shared']['roster_size']

    def collect_trajectory(self, policy, active_agents_indices, num_episodes):
        '''
        Perform rollout with noisy version of policy and update replay buffer.
        '''
        for i in range(num_episodes):

            ep_traj, agg_glob_rew = self.interface.rollout(policy, active_agents_indices=active_agents_indices, noisy_action=True, noise_std=0.7)
            
            for agent_idx in active_agents_indices:
                for transition in ep_traj[agent_idx]:
                    self.replay_buffers[agent_idx].add(transition)


    def update_params(self, roster: MultiHeadActor, active_agents_indices: list, num_episodes=80, num_samples=100):
        # perform rollouts with noisy version of this policy and update the replay buffer with experiences
        self.collect_trajectory(policy=roster, active_agents_indices=active_agents_indices, num_episodes=num_episodes)

        for agent_idx in active_agents_indices:
            sampled_trans = self.replay_buffers[agent_idx].sample_transitions(num_samples=num_samples)
            if sampled_trans is None:
                return

            y_vals = []
            main_critic_predicted_vals = []

            self.main_critics[agent_idx].zero_grad() # clearing previous gradient calcs
            
            for transition in sampled_trans:
                # rep buff now has all the experiences for us to use
                with torch.no_grad():
                    next_state_target_action = self.target_policy.clean_action(transition["next_state"], agent_idx)
                    next_state_target_critic = self.target_critics[agent_idx].forward(transition["next_state"], next_state_target_action)
                    y = transition["local_reward"] + self.discount * (1 - transition["done"]) * next_state_target_critic
                    y_vals.append(y)
                
                # exiting no grad since we're gonna do gradient update for this nn
                curr_state_main_critic = self.main_critics[agent_idx].forward(transition["state"], transition["action"])
                main_critic_predicted_vals.append(curr_state_main_critic)
            
            # gradient update for main critic
            main_critic_predicted_vals = torch.cat(main_critic_predicted_vals)
            y_vals = torch.cat(y_vals)
            main_critic_loss = criterion(main_critic_predicted_vals, y_vals)

            # Zero optimizer gradients
            self.main_critic_optims[agent_idx].zero_grad()

            main_critic_loss.backward()
            self.main_critic_optims[agent_idx].step()

            # gradient update for main policy (roster)
            roster.zero_grad()

            main_policy_vals = []
            curr_states = []
            for transition in sampled_trans:
                curr_states.append(transition["state"])
                main_policy_curr_action = roster.clean_action(transition["state"], agent_idx)
                main_policy_vals.append(main_policy_curr_action)
            
            main_policy_vals = torch.stack(main_policy_vals)
            curr_states = torch.stack(curr_states)

            # Freeze critic parameters during actor update
            for param in self.main_critics[agent_idx].parameters():
                param.requires_grad = False

            main_policy_loss = -self.main_critics[agent_idx].forward(curr_states, main_policy_vals)
            main_policy_loss = main_policy_loss.mean()

            # Zero optimizer gradients
            self.main_policy_optim.zero_grad()

            main_policy_loss.backward()
            self.main_policy_optim.step()

            # Unfreeze critic parameters after actor update
            for param in self.main_critics[agent_idx].parameters():
                param.requires_grad = True
            
            print("Agent: ", agent_idx, "Main Policy Loss", main_policy_loss.item(), "Critic Loss", main_critic_loss.item())

            soft_update(self.target_critics[agent_idx], self.main_critics[agent_idx], self.tau)
        
        # Soft update the target policy after roster has been updated using all agents' experiences
        soft_update(self.target_policy, roster, self.tau)

        # print(self.interface.rollout(self.main_policy, [0]))

if __name__ == "__main__":
    mha = MultiHeadActor(10, 2, 125, 2)
    ddpg = DDPG("/home/thakarr/IJCAI25/MOMERL/config/MARMOTConfig.yaml", "/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml", init_target_policy=mha)
    for i in range(3000):
        print("Epoch:", i)
        ddpg.update_params(roster=mha, active_agents_indices=[0, 1], num_episodes=25, num_samples=250)

    print(ddpg.interface.rollout(mha, [0, 1], alg="ddpg", noisy_action=False))