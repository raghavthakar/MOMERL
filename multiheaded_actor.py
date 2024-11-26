import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadActor(nn.Module):
    """
    Multi-headed actor network for DDPG.
    Each head represents actions for a specific policy or agent.
    """

    def __init__(self, num_state_inputs, num_actions, hidden_size, num_heads):
        super(MultiHeadActor, self).__init__()

        self.num_heads = num_heads
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # Shared trunk layers
        self.linear1 = nn.Linear(num_state_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Output layer for all heads combined
        self.mean = nn.Linear(hidden_size, num_actions * num_heads)

        # Log standard deviation as a learnable parameter for each action dimension and head
        self.log_std = nn.Parameter(torch.zeros(num_actions * num_heads))

        # Initialize weights
        self.apply(self._weights_init)

        self.id = mha_id

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
            return mean_output[start:end]
        else:
            raise IndexError(f"Head index {head} is out of range for {self.num_heads} heads.")
        

    def select_action(self, state, head=-1):
        """
        Sample stochastic actions for a given state using a specified policy head.

        Parameters:
            state (tensor): The input state.
            head (int, optional): Index of the head to use for action selection.

        Returns:
            action (tensor): The sampled action.
            action_log_prob (tensor): The log probability of the action.
        """

        if(head < 0 or head >= self.num_heads):
            raise ValueError("Head index must be within range!")

        x = self.forward(state)
        mean_output = self.mean(x)

        # Extract mean and log_std for the selected head
        start = head * self.num_actions
        end = start + self.num_actions
        mean = mean_output[:, start:end]
        log_std = self.log_std[start:end]  # Parameter tensor
        std = torch.exp(log_std)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample action
        action = dist.rsample()  # Reparameterization trick

        # Compute log probability
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Apply tanh to bound actions between -1 and 1
        action = torch.tanh(action)

        return action, action_log_prob
    

    def get_log_prob(self, state, action, head=-1):
        """
        Compute log probability and entropy of the given action under the policy.

        Parameters:
            state (tensor): The input state.
            action (tensor): The action taken.
            head (int, optional): Index of the head.

        Returns:
            action_log_prob (tensor): The log probability of the action.
            entropy (tensor): The entropy of the action distribution.
        """

        if(head < 0 or head >= self.num_heads):
            raise ValueError("Head index must be within range!")

        x = self.forward(state)
        mean_output = self.mean(x)

        # Extract mean and log_std for the selected head
        start = head * self.num_actions
        end = start + self.num_actions
        mean = mean_output[:, start:end]
        log_std = self.log_std[start:end]  # Parameter tensor
        std = torch.exp(log_std)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh transformation
        action = torch.atanh(action.clamp(-0.999, 0.999))

        # Compute log probability
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action_log_prob, entropy


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
