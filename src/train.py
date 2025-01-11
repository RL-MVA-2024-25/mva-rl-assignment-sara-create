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
import torch.nn.functional as F
from collections import namedtuple, deque
import math


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ProjectAgent:
    def __init__(self):
        """
        Initialize the agent.
        
        Args:
        - n_observations: Number of input features (state size).
        - n_actions: Number of possible actions.
        - model_path: Path to save/load the model.
        """

        self.env = env
        self.device = device
        self.n_actions  = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]

        # Hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr =1e-4

        # Networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        # Epsilon-greedy tracking
        self.steps_done = 0

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()



    def train(self, n_episodes=1000):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for i_episode in range(n_episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in range(200):
                action = self.act(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

            
    def act(self, observation, use_random=False): 
        """
        Select an action based on the observation.
        
        Args:
        - observation: Current state.
        - use_random: Whether to select a random action.
        
        Returns:
        - action: Selected action.
        """
        self.steps_done += 1
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        if use_random or random.random() < self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay):
            action =  torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(observation).max(1)[1].view(1, 1)

        return action
    
    def load(self):
        """
        Load the model from the model_path.
        """
        self.policy_net.load_state_dict(torch.load('model.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self,path):
        """
        Save the model to the model_path.
        """
        torch.save(self.policy_net.state_dict(), 'model.pth')
        

if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(n_episodes=10)
    agent.save('model.pth')
    print("Model saved.")
         
        
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
"""
