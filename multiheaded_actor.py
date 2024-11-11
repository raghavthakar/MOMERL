import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MultiHeadActor(nn.Module):
    """
    Actor model

        Parameters:
            args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions, hidden_size, num_heads):
        super(MultiHeadActor, self).__init__()

        self.num_heads = num_heads
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # Trunk
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Heads
        self.mean = nn.Linear(hidden_size, num_actions * num_heads)
        self.noise = torch.Tensor(num_actions * num_heads)

        self.apply(self.weights_init_policy_fn)

    def clean_action(self, state, head=-1):
        """
        Method to forward propagate through the actor's graph

        Parameters:
                input (tensor): state
                input (int): head_number

        Returns:
                action (tensor): actions
        """

        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        mean = torch.tanh(self.mean(x))

        if head == -1:
            return mean # return the whole output layer if head is not specified
        
        elif head >= len(mean) / self.num_actions:
            print("[multiheaded_actor: Index for head in forward pass must not be greater than number of heads. Exiting...")
            exit()

        else:
            start = head * self.num_actions
            return mean[:, start:start + self.num_actions]

    def noisy_action(self, state, head=-1):
        # print("State: ", state)

        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        mean = torch.tanh(self.mean(x))

        action = mean + self.noise.normal_(0., std=0.4)

        if head == -1:
            return action

        else:
            start = head * self.num_actions

            return action[:, start:start + self.num_actions]

    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])

        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means) / len(means)

        return minimum, maximum, mean

    # Initialize Policy weights
    def weights_init_policy_fn(self, m):
        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

if __name__ == "__main__":

    mha = MultiHeadActor(2, 2, 3, 3)
    
    for name, param in mha.named_parameters():
        if param.requires_grad:  # Only print parameters that require gradients (i.e., weights and biases)
            print(f"Layer: {name}, Shape: {param.shape}, Weights: {param.data}")
    
