import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random
import yaml

from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface
from multiheaded_actor import MultiHeadActor
from DDPGCritic import Critic
from ReplayBuffer import ReplayBuffer

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
    def __init__(self, alg_config_filename, rover_config_filename):
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
            hard_update(self.target_policies[i], self.main_policies[i])

            self.main_critic_optims.append(torch.optim.Adam(self.main_critic.parameters(), lr=self.critic_lr))

            self.replay_buffers.append(ReplayBuffer(10000))
        
        # one main and target policy
        self.main_policy = MultiHeadActor(self.interface.get_state_size(), self.interface.get_action_size(), self.actor_hidden_size, self.roster_size)
        self.target_policy = MultiHeadActor(self.interface.get_state_size(), self.interface.get_action_size(), self.actor_hidden_size, self.roster_size)
        self.main_policy_optim = torch.optim.Adam(self.main_policy.parameters(), lr=self.actor_lr)
    
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

    
    def collect_trajectory(self, policy, active_agents_indices, num_episodes, num_samples):
        for i in range(num_episodes):

            ep_traj, agg_glob_rew = self.interface.rollout(policy, active_agents_indices=active_agents_indices, noisy_action=True, noise_std=0.7)
            
            for agent_idx in active_agents_indices:
                for transition in ep_traj[agent_idx]:
                    self.replay_buffers[agent_idx].add(transition)

        sampled_transitions = np.random.choice(self.rep_buff.experiences, size=num_samples, replace=False)
        return sampled_transitions

    def update_params(self, num_epochs=500, num_episodes=80, num_samples=100):

        for e in range(num_epochs):
            #self.rep_buff.experiences = []

            sampled_trans = self.collect_trajectory(num_episodes, num_samples)
            #sampled_trans = self.gather_and_sample_transitions(num_episodes, num_samples)


            y_vals = []
            main_critic_preds = []

            self.main_critic.zero_grad() # clearing previous gradient calcs
            for transition in sampled_trans:
                # rep buff now has all the experiences for us to use
                # next_state_target_pol = []
                with torch.no_grad():
                    next_state_target_pol = self.target_policy.clean_action(transition["next_state"], 0)
                    
                    next_state_target_critic = self.target_critic.forward(transition["next_state"], next_state_target_pol)
                    
                    y = transition["local_reward"] + self.discount * (1 - transition["done"]) * next_state_target_critic
                    y_vals.append(y)
                    #print(y, transition["local_reward"], transition["done"], next_state_target_critic)
                
                # exiting no grad since we're gonna do gradient update for this nn
                curr_state_main_critic = self.main_critic.forward(transition["state"], transition["action"])
                #print(curr_state_main_critic)
                main_critic_preds.append(curr_state_main_critic)
            
            # gradient update for main critic
            main_critic_preds = torch.cat(main_critic_preds)
            y_vals = torch.cat(y_vals)
            main_critic_loss = criterion(main_critic_preds, y_vals)
            main_critic_loss.backward()
            self.optim_main_critic.step()

            # gradient update for main policy
            self.main_policy.zero_grad()

            main_policy_vals = []
            curr_state_lst = []
            for transition in sampled_trans:
                curr_state_lst.append(transition["state"])
                main_policy_curr_state = self.main_policy.clean_action(transition["state"], 0)
                main_policy_vals.append(main_policy_curr_state)
            
            main_policy_vals = torch.stack(main_policy_vals)
            curr_state_lst = torch.stack(curr_state_lst)
            #main_policy_vals = main_policy_vals.unsqueeze(0)
            
            # TODO: Check if this messes up the gradient since it updates the main critic too (is this still true? need to check)

            

            main_policy_loss = -self.main_critic.forward(curr_state_lst, main_policy_vals)
            main_policy_loss = main_policy_loss.mean()
            main_policy_loss.backward()
            self.optim_main_policy.step()
            print("Epoch", e, "Main Policy Loss", main_policy_loss.item(), "Critic Loss", main_critic_loss.item())

            soft_update(self.target_critic, self.main_critic, self.tau)
            soft_update(self.target_policy, self.main_policy, self.tau)

        print(self.interface.rollout(self.main_policy, [0]))

if __name__ == "__main__":
    ddpg = DDPG("/home/thakarr/IJCAI25/MOMERL/config/MARMOTConfig.yaml", "/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml")
    ddpg.update_params(3000, 25, 250)
    # ddpg.update_params(1, 1, 100)