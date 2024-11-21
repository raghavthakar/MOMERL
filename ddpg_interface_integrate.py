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
            target_param.data * (1.0 - tau) + param.data * tau
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
    def __init__(self, buff_size=10000):
        self.experiences = []
        self.buff_size = buff_size
    
    def add(self, transition, random_discard=True):
        if len(self.experiences) < self.buff_size:
            self.experiences.append(transition)
        else:
            if random_discard:
                i = random.randrange(len(self.experiences)) # get random index
                self.experiences[i], self.experiences[0] = self.experiences[0], self.experiences[i] # swap with the first element

            self.experiences.pop(0)
            self.experiences.append(transition)

class DDPG2:
    def __init__(self, rover_config_filename):
        # initialize main critic, target critic, main policy, target policy
        self.main_critic = Critic(10, 2, 25)
        self.target_critic = Critic(10, 2, 25)

        self.main_policy = MultiHeadActor(10, 2, 125, 1)
        self.target_policy = MultiHeadActor(10, 2, 125, 1)

        hard_update(self.target_critic, self.main_critic)
        hard_update(self.target_policy, self.main_policy)

        self.optim_main_critic = torch.optim.Adam(self.main_critic.parameters(), lr=0.001)
        self.optim_main_policy = torch.optim.Adam(self.main_policy.parameters(), lr=0.0005)
        self.interface = MORoverInterface(rover_config_filename)

        self.rep_buff = ReplayBuffer(10000)

        self.tau = 0.005
        self.discount = 0.999
    
    def collect_trajectory(self, num_episodes, num_samples):
        for i in range(num_episodes):

            '''
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
                noise = torch.normal(mean=0.0, std=0.7, size=action.size())
                action += noise
                
                action = torch.clamp(action, -1.0, 1.0)
                action = action.detach().numpy()[0]

                # Scale the action to comply with the agent's max step size
                norm = np.linalg.norm(action) # get the magnitude of the calculated move
                max_step = max_step_sizes[0]
                scaling_factor = (max_step / norm) if norm > 0 else 0 # the factor by which the moves should be scaled
                scaled_action = action * scaling_factor # multiply each member of the action by the scaling factor

                # print(action)
                # print(agent_locations)
                # [ 0.4702882 -0.1327428]
                # [[np.float32(6.6230736), np.float32(2.7077267)]]

                agent_locations = self.interface.rover_env.update_agent_locations(
                    agent_locations, [scaled_action], max_step_sizes)

                next_observations_list = self.interface.rover_env.generate_observations(
                    agent_locations, num_sensors, observation_radii, normalise=True)
                next_state = next_observations_list[0]

                local_rewards = self.interface.rover_env.get_local_rewards(agent_locations)
                reward = local_rewards[0]

                done = (t == ep_length - 1)

                state = np.array(state, dtype=np.float32)
                state = torch.tensor(state, dtype=torch.float32, requires_grad=True)

                action = np.array(action, dtype=np.float32)
                action = torch.tensor(action, dtype=torch.float32, requires_grad=True)

                next_state = np.array(next_state, dtype=np.float32)
                next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=True)

                single_trans = {"state": state, "action": action, "local_reward": reward, "next_state": next_state, "done": done}
                
                self.rep_buff.add(single_trans)

                if done:
                    break

        '''

            ep_traj, agg_glob_rew = self.interface.rollout(self.main_policy, [0], noisy_action=True, noise_std=0.7)
            
            for transition in ep_traj[0]:
                self.rep_buff.add(transition)

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
    ddpg = DDPG2("/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml")
    ddpg.update_params(3000, 25, 250)
    # ddpg.update_params(1, 1, 100)