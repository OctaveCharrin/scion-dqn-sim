import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class SimpleDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimpleDQNNetwork, self).__init__()
        # As per the paper: "Typically, this is implemented as a multi-layer perceptron (MLP)"
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class SimpleDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=32, target_update_every=100, min_buffer_size=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.min_buffer_size = min_buffer_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = SimpleDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = SimpleDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Standard Experience Replay Buffer
        self.memory = deque(maxlen=buffer_size)
        self.steps = 0
        self.episodes = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.min_buffer_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q(s', a') using Target Network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        # MSE Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
