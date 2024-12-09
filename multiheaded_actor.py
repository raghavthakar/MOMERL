import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadActor(nn.Module):
    """
    Multi-headed actor network with per-head hidden layers.
    Each head represents actions for a specific agent.
    """

    def __init__(self, num_state_inputs, num_actions, hidden_size, num_heads):
        super(MultiHeadActor, self).__init__()

        self.num_heads = num_heads
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # Shared trunk layer
        self.linear1 = nn.Linear(num_state_inputs, self.hidden_size)

        # Per-head layers: Each head has its own hidden layer and output layer
        self.per_head_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//4),
                nn.ReLU(),
                nn.Linear(self.hidden_size//4, self.hidden_size//4),
                nn.ReLU(),
                nn.Linear(self.hidden_size//4, self.num_actions)
            ) for _ in range(self.num_heads)
        ])

        self.reset_parameters()

    def forward(self, state, head=-1):
        """
        Forward pass through the network.
        """
        # Shared layers
        x = F.relu(self.linear1(state))

        if head == -1:
            # Return actions from all heads
            outputs = []
            for per_head_layer in self.per_head_layers:
                head_output = torch.tanh(per_head_layer(x))
                outputs.append(head_output)
            return torch.cat(outputs, dim=1)
        elif 0 <= head < self.num_heads:
            # Return action from the specified head
            head_output = torch.tanh(self.per_head_layers[head](x))
            return head_output
        else:
            raise IndexError(f"Head index {head} is out of range for {self.num_heads} heads.")

    def clean_action(self, state, head=-1):
        """
        Generate deterministic actions for a given state using a specified policy head (or all heads).
        """
        return self.forward(state, head=head)

    def reset_parameters(self):
        # Initialize shared layers
        nn.init.uniform_(self.linear1.weight, -0.2, 0.2)
        nn.init.uniform_(self.linear1.bias, -0.0, 0.0)

        # Initialize per-head layers
        for per_head_layer in self.per_head_layers:
            nn.init.uniform_(per_head_layer[0].weight, -0.2, 0.2)
            nn.init.uniform_(per_head_layer[0].bias, -0.0, 0.0)
            nn.init.uniform_(per_head_layer[2].weight, -0.2, 0.2)
            nn.init.uniform_(per_head_layer[2].bias, -0.0, 0.0)
            nn.init.uniform_(per_head_layer[4].weight, -0.02, 0.02)  # Smaller init for output layer
            nn.init.uniform_(per_head_layer[4].bias, -0.0, 0.0)

    def mutate(self, noise_mean, noise_std):
        """
        Mutate the network by adding noise to the weights.
        """
        with torch.no_grad():
            # Mutate shared layers
            for layer in [self.linear1]:
                if hasattr(layer, 'weight'):
                    layer.weight.add_(torch.randn_like(layer.weight) * noise_std + noise_mean)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.add_(torch.randn_like(layer.bias) * noise_std + noise_mean)

            # Mutate per-head layers
            for per_head_layer in self.per_head_layers:
                for sub_layer in per_head_layer:
                    if hasattr(sub_layer, 'weight'):
                        sub_layer.weight.add_(torch.randn_like(sub_layer.weight) * noise_std + noise_mean)
                    if hasattr(sub_layer, 'bias') and sub_layer.bias is not None:
                        sub_layer.bias.add_(torch.randn_like(sub_layer.bias) * noise_std + noise_mean)
