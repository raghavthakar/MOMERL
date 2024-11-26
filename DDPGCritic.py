import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, num_state_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_state_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(self._weights_init_value_fn)

        self.loss_fn = nn.MSELoss()
    
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