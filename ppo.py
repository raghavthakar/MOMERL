# ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from MORoverEnv import MORoverEnv
from MORoverInterface import MORoverInterface

class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PPOActorCritic, self).__init__()

        # Actor network
        self.actor_fc1 = nn.Linear(num_inputs, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_output = nn.Linear(hidden_size, num_actions)

        # Critic network
        self.critic_fc1 = nn.Linear(num_inputs, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_output = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._weights_init)

    def forward(self, state):
        # Actor forward pass
        x = torch.relu(self.actor_fc1(state))
        x = torch.relu(self.actor_fc2(x))
        action_mean = self.actor_output(x)
        action_dist = torch.tanh(action_mean)  # Ensure actions are in [-1, 1]
        
        # Critic forward pass
        v = torch.relu(self.critic_fc1(state))
        v = torch.relu(self.critic_fc2(v))
        state_value = self.critic_output(v)

        return action_dist, state_value

    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

class PPOAgent:
    def __init__(self, rover_config_filename, hidden_size=128, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, update_epochs=10, gae_lambda=0.95, 
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5):
        self.interface = MORoverInterface(rover_config_filename)
        with open(rover_config_filename, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.state_dim = self._get_state_dim(config)
        self.action_dim = self._get_action_dim(config)
        self.hidden_size = hidden_size

        self.actor_critic = PPOActorCritic(self.state_dim, self.action_dim, hidden_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def _get_state_dim(self, config):
        num_sensors = config['Agents']['num_sensors'][0]
        num_dims = len(config['Environment']['dimensions'])
        # For 2D: normalized position (2) + POI observations + agent observations
        state_dim = num_dims + num_sensors * 2
        return state_dim

    def _get_action_dim(self, config):
        num_dims = len(config['Environment']['dimensions'])
        return num_dims

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.actor_critic(state)
        action_dist = torch.distributions.Normal(action_mean, torch.tensor(0.1))
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], log_prob.detach().numpy()[0]

    def train(self, num_episodes=1000, batch_size=128, timesteps_per_batch=2048):
        total_steps = 0
        for episode in range(num_episodes):
            trajectories = []
            total_rewards = []
            steps = 0

            while steps < timesteps_per_batch:
                trajectory = self.collect_trajectory()
                trajectories.append(trajectory)
                steps += len(trajectory['rewards'])
                total_rewards.append(sum(trajectory['rewards']))

            total_steps += steps

            # Flatten the trajectories
            states = np.concatenate([t['states'] for t in trajectories])
            actions = np.concatenate([t['actions'] for t in trajectories])
            log_probs = np.concatenate([t['log_probs'] for t in trajectories])
            rewards = np.concatenate([t['rewards'] for t in trajectories])
            dones = np.concatenate([t['dones'] for t in trajectories])
            next_states = np.concatenate([t['next_states'] for t in trajectories])

            # Compute advantages and returns
            advantages, returns = self.compute_gae(states, rewards, dones, next_states)

            # Convert to tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            log_probs_tensor = torch.FloatTensor(log_probs)
            advantages_tensor = torch.FloatTensor(advantages)
            returns_tensor = torch.FloatTensor(returns)

            # Update policy
            self.ppo_update(states_tensor, actions_tensor, log_probs_tensor, advantages_tensor, returns_tensor)

            # Logging
            avg_reward = np.mean(total_rewards)
            print(f"Episode {episode+1}, Steps {total_steps}, Average Reward: {avg_reward:.2f}")

    def collect_trajectory(self):
        trajectory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'dones': [], 'next_states': []}
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

            action, log_prob = self.select_action(state)
            action = action * max_step_sizes[0]  # Scale action

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
            trajectory['log_probs'].append(log_prob)
            trajectory['dones'].append(done)
            trajectory['next_states'].append(next_state)

            if done:
                break

        return trajectory

    def compute_gae(self, states, rewards, dones, next_states):
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)

        with torch.no_grad():
            _, state_values = self.actor_critic(states_tensor)
            _, next_state_values = self.actor_critic(next_states_tensor)

        state_values = state_values.squeeze().numpy()
        next_state_values = next_state_values.squeeze().numpy()

        deltas = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        advantages = []
        advantage = 0.0
        for delta, done in zip(reversed(deltas), reversed(dones)):
            if done:
                advantage = 0.0
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)
        advantages = np.array(advantages)
        returns = advantages + state_values
        return advantages, returns

    def ppo_update(self, states, actions, log_probs_old, advantages, returns):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.update_epochs):
            # Get new log probs and state values
            action_means, state_values = self.actor_critic(states)
            action_dist = torch.distributions.Normal(action_means, torch.tensor(0.1))
            log_probs = action_dist.log_prob(actions).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.value_loss_coef * (returns - state_values.squeeze()).pow(2).mean()
            entropy_loss = -self.entropy_coef * entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

if __name__ == "__main__":
    ppo_agent = PPOAgent(
        rover_config_filename='/home/thakarr/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml',
        hidden_size=128,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        update_epochs=10,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5
    )
    ppo_agent.train(num_episodes=1000, batch_size=128, timesteps_per_batch=2048)
