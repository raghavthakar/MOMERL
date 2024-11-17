import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random

from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface
from multiheaded_actor import MultiHeadActor

criterion = nn.MSELoss()

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# TODO: check to see if this soft update is correct or not
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            param.data * (1.0 - tau) + target_param.data * tau
        )

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(self._weights_init_value_fn)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    # Initialize Policy weights
    def _weights_init_value_fn(self, m):
        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        x1 = torch.cat([state, action], -1) # TODO: changed from 0 back to 1
        x1 = torch.tanh(self.linear1(x1))
        x1 = torch.tanh(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

class ReplayBuffer:
    def __init__(self):
        self.experiences = []
        self.buff_size = 10000

class DDPG2:
    def __init__(self, rover_config_filename):
        # initialize main critic, target critic, main policy, target policy
        self.main_critic = Critic(10, 2, 25)
        self.target_critic = Critic(10, 2, 25)

        self.main_policy = MultiHeadActor(10, 2, 125, 1)
        self.target_policy = MultiHeadActor(10, 2, 125, 1)

        hard_update(self.target_critic, self.main_critic)
        hard_update(self.target_critic, self.main_critic)

        self.optim_main_critic = torch.optim.Adam(self.main_critic.parameters(), lr=0.001)
        self.optim_main_policy = torch.optim.Adam(self.main_policy.parameters(), lr=0.00001)
        self.interface = MORoverInterface(rover_config_filename)

        self.rep_buff = ReplayBuffer()

        self.tau = 0.001
        self.discount = 0.99
    
    def collect_trajectory(self, num_episodes, num_samples):
        for i in range(num_episodes):
            trajectory = {'states': [], 'actions': [], 'rewards': [], 'done': [], 'next_states': []}
            agent_indices = [0]  # Assuming single agent
            agent_locations = self.interface.config['Agents']['starting_locs']
            num_sensors = self.interface.config['Agents']['num_sensors']
            observation_radii = self.interface.config['Agents']['observation_radii']
            max_step_sizes = self.interface.config['Agents']['max_step_sizes']
            ep_length = self.interface.rover_env.get_ep_length()

            self.interface.rover_env.reset()
            for t in range(ep_length):
                observations_list = self.interface.rover_env.generate_observations(
                    agent_locations, num_sensors, observation_radii, normalise=True)
                state = observations_list[0]

                action = self.main_policy.clean_action(torch.FloatTensor(state).unsqueeze(0))
                noise = torch.normal(mean=0.0, std=3.0, size=action.size())
                action += noise
                action = action * max_step_sizes[0]  # Scale action
                action = action.detach().numpy()[0]

                # print(action)
                # print(agent_locations)
                # [ 0.4702882 -0.1327428]
                # [[np.float32(6.6230736), np.float32(2.7077267)]]

                agent_locations = self.interface.rover_env.update_agent_locations(
                    agent_locations, [action], max_step_sizes)

                next_observations_list = self.interface.rover_env.generate_observations(
                    agent_locations, num_sensors, observation_radii, normalise=True)
                next_state = next_observations_list[0]

                local_rewards = self.interface.rover_env.get_local_rewards(agent_locations)
                reward = local_rewards[0]

                done = (t == ep_length - 1)

                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['done'].append(done)
                trajectory['next_states'].append(next_state)

                # {'state': [np.float64(0.187927508354187), np.float64(0.3044823169708252), 1, 0, 0, 0, 0, 0, 0, 0], 'action': array([ 0.03726111, -0.6031788 ], dtype=float32), 'local_reward': 2.9230006696420846e-15, 'next_state': [np.float64(0.19409321546554564), np.float64(0.20467257499694824), 1, 0, 0, 0, 0, 0, 0, 0], 'done': False}
                # {'state': tensor([0.1879, 0.3045, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                #         0.0000], requires_grad=True), 'action': tensor([ 0.0373, -0.6032], requires_grad=True), 'local_reward': 2.9230006696420846e-15, 'next_state': tensor([0.1941, 0.2047, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                #         0.0000], requires_grad=True), 'done': False}

                state = np.array(state, dtype=np.float32)
                state = torch.tensor(state, dtype=torch.float32, requires_grad=True)

                action = np.array(action, dtype=np.float32)
                action = torch.tensor(action, dtype=torch.float32, requires_grad=True)

                next_state = np.array(next_state, dtype=np.float32)
                next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=True)

                single_trans = {"state": state, "action": action, "local_reward": reward, "next_state": next_state, "done": done}
                
                if(len(self.rep_buff.experiences) < self.rep_buff.buff_size):
                    self.rep_buff.experiences.append(single_trans)
                # print(trans)
                # print()
                else:
                    self.rep_buff.experiences.pop(0)
                    self.rep_buff.experiences.append(single_trans)

                if done:
                    break
            
            # print(trajectory)
            # print()

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
                    next_state_target_pol = self.target_policy.clean_action(transition["next_state"])
                    
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
                main_policy_curr_state = self.main_policy.clean_action(transition["state"])
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
    ddpg = DDPG2("/Users/sidd/Desktop/ijcai25/even_newer_MOMERL/MOMERL/config/MORoverEnvConfig.yaml")
    ddpg.update_params(3000, 25, 100)