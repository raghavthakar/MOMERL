import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random

from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface
from multiheaded_actor import MultiHeadActor


class Critic(nn.Module):
    """
    Critic model

        Parameters:
            args (object): Parameter class
    """
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(self._weights_init_value_fn)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
    
    # Initialize Policy weights
    def _weights_init_value_fn(self, m):
        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = torch.tanh(self.linear1(x1))
        x1 = torch.tanh(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1
    
class ReplayBuffer:
    def __init__(self, buff_size=200):
        self.experiences = []
        self.buff_size = buff_size

class DDPG:
    def __init__(self, alg_config_filename, rover_config_filename):
        self.gamma = 0.1
        self.main_policy = MultiHeadActor(10, 2, 5, 1, 1)
        self.target_policy = deepcopy(self.main_policy)
        self.main_critic = Critic(10, 2, 25)
        self.target_critic = deepcopy(self.main_critic)
        self.rep_buf = ReplayBuffer()
        self.interface = MORoverInterface(rover_config_filename)

    def _soft_update(self, target_network, main_network, tau):
        """
        Performs a soft update of the target network parameters by blending them with the main network parameters.
        """
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
        
    def run(self, tau=0.01):
        for e in range(250):
            self.rep_buf.experiences.clear()

            for _ in range(200):
                experience, glob_reward = self.interface.rollout(self.main_policy, [0], True)
                ep_traj = experience[0]
                for transition in ep_traj:
                    self.rep_buf.experiences.append(transition)
            
            # finished filling replay buffer

            # randomly sample n transitions from self.rep_buf.experiences
            if len(self.rep_buf.experiences) < 100:
                # Not enough data for a batch
                continue
            
            sampled_transitions = np.random.choice(self.rep_buf.experiences, size=30, replace=False)
            # print(choices)

            main_q_vals_exp, y_list = [], []
       
            for transition in sampled_transitions:
                state_tensor = torch.Tensor(transition['state'])
                next_state_tensor = torch.Tensor(transition['next_state'])
                action_tensor = torch.Tensor([transition['action']])

                target_next_action = self.target_policy.clean_action(next_state_tensor)
                target_q_value = self.target_critic.forward(state=next_state_tensor, action=target_next_action)

                y = transition['local_reward'] + self.gamma * (1 - int(transition['done'])) * float(target_q_value[0])
                y_list.append(y)

                main_q_value_exp = self.main_critic.forward(state=state_tensor, action=action_tensor)
                main_q_vals_exp.append(main_q_value_exp)

            
            # update the critic
            main_critic_loss = self.main_critic.loss_fn(torch.stack(main_q_vals_exp), torch.tensor(y_list).unsqueeze(1))
            self.main_critic.optimizer.zero_grad()
            main_critic_loss.backward()
            self.main_critic.optimizer.step()
            # print("Main Critic Loss", main_critic_loss.item())

            # update the actor
            main_q_vals_main = []
            for transition in sampled_transitions:
                state_tensor = torch.Tensor(transition['state'])
                predicted_action = self.main_policy.clean_action(state_tensor)
                q_val = self.main_critic.forward(state=state_tensor, action=predicted_action)
                main_q_vals_main.append(q_val)

            policy_loss = -torch.stack(main_q_vals_main).mean()
            self.main_policy.optimizer.zero_grad()
            policy_loss.backward()
            self.main_policy.optimizer.step()
            print("Main Policy Loss", policy_loss.item())

            # perform a soft update of target networks
            self._soft_update(self.target_policy, self.main_policy, tau)
            self._soft_update(self.target_critic, self.main_critic, tau)

        print(self.interface.rollout(self.main_policy, [0]))


if __name__ == "__main__":
    ddpg = DDPG(None, '/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml')
    ddpg.run(tau=0.001)
