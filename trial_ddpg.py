import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface
from multiheaded_actor import MultiHeadActor

class Critic(nn.Module):
    """
    Critic model for DDPG that estimates the value (Q-function) of a given state-action pair.
    """

    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # Critic architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(self._weights_init_value_fn)
        self.loss_fn = nn.MSELoss()

    def forward(self, state, action):
        """
        Forward pass of the Critic.

        Parameters:
            state (Tensor): The input state.
            action (Tensor): The input action.

        Returns:
            Tensor: The estimated Q-value.
        """
        x = torch.cat([state, action], 1)  # Combine state and action
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def _weights_init_value_fn(self, layer):
        """
        Initializes weights for value network layers using Kaiming initialization for ReLU.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

class ReplayBuffer:
    """
    Replay Buffer to store tuples of experiences for training DDPG.
    """

    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        """
        Add an experience tuple to the replay buffer.
        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove oldest experience if buffer is full
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the replay buffer.
        """
        return np.random.choice(self.buffer, size=batch_size, replace=False)

    def clear(self):
        """
        Clear the replay buffer.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) implementation suitable for single-agent continuous control,
    adapted for multi-headed actor networks.
    """

    def __init__(self, alg_config_filename, rover_config_filename, state_dim, action_dim, hidden_size=128, gamma=0.99, tau=0.001, actor_lr=1e-3, critic_lr=1e-3):
        self.gamma = gamma
        self.tau = tau

        # Initialize main and target networks
        self.main_actor = MultiHeadActor(num_inputs=state_dim, num_actions=action_dim, hidden_size=hidden_size, num_heads=1)
        self.target_actor = deepcopy(self.main_actor)

        self.main_critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        self.target_critic = deepcopy(self.main_critic)

        # Set up optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=critic_lr)

        # Replay buffer for experiences
        self.replay_buffer = ReplayBuffer()

        # Environment interface
        self.interface = MORoverInterface(rover_config_filename)

    def _soft_update(self, target_network, main_network):
        """
        Perform a soft update of the target network parameters by blending them with the main network parameters.
        """
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, episodes=500, rollouts_per_episode=20, batch_size=64, secondary_update_freq=5):
        """
        Train the DDPG agent using experiences collected from the environment.
        """
        for e in range(episodes):
            # Clear the replay buffer for each episode
            self.replay_buffer.clear()

            # Collect experiences by performing rollouts in the environment
            for _ in range(rollouts_per_episode):
                experience, global_reward = self.interface.rollout(self.main_actor, active_agents_indices=[0], noisy_action=True, noise_std=0.05)
                # Since we have only one agent (head=0) in this scenario
                agent_trajectory = experience[0]

                for transition in agent_trajectory:
                    self.replay_buffer.add(transition)

            if len(self.replay_buffer) < batch_size:
                # Not enough data for a batch
                continue

            # Sample a batch of experiences from the replay buffer
            sampled_transitions = self.replay_buffer.sample(batch_size)

            # Convert the batch of experiences to tensors
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for transition in sampled_transitions:
                states.append(transition['state'])
                actions.append(transition['action'])
                rewards.append([transition['local_reward']])
                next_states.append(transition['next_state'])
                dones.append([float(transition['done'])])

            state_tensor = torch.FloatTensor(np.array(states))
            action_tensor = torch.FloatTensor(np.array(actions))
            reward_tensor = torch.FloatTensor(np.array(rewards))
            next_state_tensor = torch.FloatTensor(np.array(next_states))
            done_tensor = torch.FloatTensor(np.array(dones))

            # -------- Update Critic --------
            with torch.no_grad():
                # Target actor provides actions for the next states
                target_next_action = self.target_actor.clean_action(next_state_tensor)
                # Evaluate critic for next states and actions
                target_q_value = self.target_critic(next_state_tensor, target_next_action)
                # Compute the target value y for the critic loss
                y = reward_tensor + (1 - done_tensor) * self.gamma * target_q_value

            # Compute Q-value from main_critic for the sampled state-action pairs
            main_q_value = self.main_critic(state_tensor, action_tensor)
            critic_loss = self.main_critic.loss_fn(main_q_value, y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # -------- Update Actor --------
            # The actor aims to maximize the Q-value given by main_critic
            predicted_action = self.main_actor.clean_action(state_tensor)
            actor_loss = -self.main_critic(state_tensor, predicted_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # -------- Soft Update Target Networks --------
            self._soft_update(self.target_actor, self.main_actor)
            self._soft_update(self.target_critic, self.main_critic)

            # Logging for debugging
            print(f"Episode {e+1}/{episodes}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
            print(self.interface.rollout(self.main_actor, [0], noisy_action=False)[0])

        # Evaluate final policy
        final_trajectory, final_global_reward = self.interface.rollout(self.main_actor, [0], noisy_action=False)
        print(final_trajectory)


if __name__ == "__main__":
    # Example initialization:
    # Adjust `state_dim` and `action_dim` to match your environment’s dimensionalities
    ddpg_agent = DDPG(
        alg_config_filename=None,
        rover_config_filename='/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml',
        state_dim=10,
        action_dim=2,
        hidden_size=128,
        gamma=0.99,
        tau=0.001,
        actor_lr=0.00001,
        critic_lr=1e-3
    )
    ddpg_agent.train(episodes=1000, rollouts_per_episode=10, batch_size=100, secondary_update_freq=5)
