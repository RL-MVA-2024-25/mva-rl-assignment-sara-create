

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
#import matplotlib.pyplot as plt
from collections import namedtuple, deque
import math
from evaluate import evaluate_HIV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    ) 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Declare network
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons=24

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class ProjectAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        
        nb_neurons=256 

        model = DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            #nn.SiLU(), ReLU seems to work better
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            #nn.SiLU(),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            #nn.SiLU(),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            #nn.SiLU(),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            #nn.SiLU(),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
            ).to(device)

        config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 100000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.,
            'epsilon_decay_period': 21000,
            'epsilon_delay_decay': 100,
            'batch_size': 790,
            'gradient_steps': 3,
            'update_target_strategy': 'replace',
            'update_target_freq': 400,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss()}

        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        previous_val = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')


                validation_score = evaluate_HIV(agent=self, nb_episode=1)          
                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {validation_score:.2e}")
                state, _ = env.reset()
                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(device)
                    self.save("model_new.pth")
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        self.save("model_new.pth")
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def load(self):
        path = 'model_new.pth'
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Load on CPU
        self.model.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}.")

    def save(self, path: str) -> None:
        torch.save({
            'policy_net_state_dict': self.model.state_dict(),
            'target_net_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}.")
    

    def act(self, observation, use_random=False): 
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        if use_random :
            return torch.tensor([[random.randrange(self.nb_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.model(observation).max(1)[1].view(1, 1)
    


if __name__ == "__main__":
    # Train agent
    agent = ProjectAgent()
    scores = agent.train(env, 350)