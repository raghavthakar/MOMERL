import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadActor(nn.Module):
    """
    Multi-headed actor network for DDPG.
    Each head represents actions for a specific policy or agent.
    """

    def __init__(self, num_inputs, num_actions, hidden_size, num_heads):
        super(MultiHeadActor, self).__init__()

        self.num_heads = num_heads
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # Shared trunk layers
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Output layer for all heads combined
        self.mean = nn.Linear(hidden_size, num_actions * num_heads)

        # Initialize weights
        self.apply(self._weights_init)

    def forward(self, state):
        """
        Forward pass through the shared trunk of the network.
        
        Parameters:
            state (tensor): The state input to the network.
        
        Returns:
            x (tensor): Activated outputs from the second hidden layer.
        """
        # Using ReLU activation for hidden layers for better gradient flow
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return x

    def clean_action(self, state, head=-1):
        """
        Generate deterministic actions for a given state using a specified policy head (or all heads).

        Parameters:
            state (tensor): The input state.
            head (int, optional): Index of the head to use for action selection. Defaults to -1,
                                  which returns actions from all heads.
        
        Returns:
            action (tensor): The computed action(s).
        """
        x = self.forward(state)
        # Scale actions to [-1, 1] range using tanh
        mean_output = torch.tanh(self.mean(x))

        if head == -1:
            # Return actions from all heads (num_heads x num_actions)
            return mean_output
        elif 0 <= head < self.num_heads:
            start = head * self.num_actions
            end = start + self.num_actions
            return mean_output[:, start:end]
        else:
            raise IndexError(f"Head index {head} is out of range for {self.num_heads} heads.")

    def _weights_init(self, layer):
        """
        Initializes weights for layers using Kaiming initialization for ReLU.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def get_norm_stats(self):
        """
        Get statistics about the norms of the network parameters. Useful for debugging.

        Returns:
            tuple: Minimum parameter value, maximum parameter value, and mean of absolute parameter values.
        """
        parameters = [p for p in self.parameters() if p.requires_grad]
        minimum = min(param.min().item() for param in parameters)
        maximum = max(param.max().item() for param in parameters)
        mean_abs_values = [torch.mean(torch.abs(param)).item() for param in parameters]
        mean = sum(mean_abs_values) / len(mean_abs_values)
        return minimum, maximum, mean
