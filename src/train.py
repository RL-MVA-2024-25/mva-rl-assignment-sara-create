import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

"""
# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectAgent:
    def __init__(self):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 0.001
        self.target_update_freq = 10  # Update target network every 10 episodes

        # Replay buffer
        self.memory = deque(maxlen=10000)

        # Networks
        self.policy_net = DQNetwork(self.state_dim, self.action_dim)
        self.target_net = DQNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.step_count = 0
        self.last_observation = None
        self.last_action = None

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    
    def act(self, observation, use_random=False):
        # Convert observation to tensor
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Select action using epsilon-greedy
        if use_random or random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)  # Random action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = torch.argmax(q_values).item()  # Greedy action

        # If this is not the first step, store the experience
        if self.last_observation is not None:
            reward = observation[0]  # Simulated reward (replace as needed)
            done = observation[1]  # Simulated done flag (replace as needed)
            self.store_experience(self.last_observation, self.last_action, reward, observation, done)
            self.train()  # Perform training step

        # Update the last observation and action
        self.last_observation = observation
        self.last_action = action

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


        return action


    def save(self, path):
        # Save the policy network and replay buffer
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)

    def load(self, path):
        # Load the policy network and replay buffer
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.buffer_size)
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

    def store_experience(self, state, action, reward, next_state, done):
        # Store a single transition in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        # Train only if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute Q-target
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, q_targets)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

"""

class ProjectAgent:
    def __init__(self, state_dim=6, action_dim=4, buffer_size=10000, batch_size=64,
                 gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, target_update_freq=10):
        # DQN parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Training step counter
        self.step_count = 0
        self.last_observation = None
        self.last_action = None

        # Default checkpoint path
        self.checkpoint_path = "agent_checkpoint.pth"

    def _build_network(self):
        # Simple feed-forward neural network
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def act(self, observation, use_random=False):
        # Convert observation to tensor
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Select action using epsilon-greedy
        if use_random or random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)  # Random action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = torch.argmax(q_values).item()  # Greedy action

        # If this is not the first step, store the experience
        if self.last_observation is not None:
            reward = observation[0]  # Simulated reward (replace as needed)
            done = observation[1]  # Simulated done flag (replace as needed)
            self.store_experience(self.last_observation, self.last_action, reward, observation, done)
            self.train()  # Perform training step

        # Update the last observation and action
        self.last_observation = observation
        self.last_action = action

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def store_experience(self, state, action, reward, next_state, done):
        # Store a single transition in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        # Train only if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute Q-target
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, q_targets)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self,path):
        # Save the policy network and replay buffer to the default path
        path = self.checkpoint_path
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)

    def load(self):
        # Load the policy network and replay buffer from the default path
        if not os.path.exists(self.checkpoint_path):
            return
        checkpoint = torch.load(self.checkpoint_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.buffer_size)
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
