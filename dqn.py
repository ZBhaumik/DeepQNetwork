import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np

class DQN:
    def __init__(self, state_size, action_size, gamma, tau, lr):
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.get_device())
        
        self.target_model = deepcopy(self.model).to(self.get_device()) # Copy initial weights to target model.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(0, self.action_size)
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(self.get_device())
            return torch.argmax(self.model(state)).item()

    def update(self, batch, weights=None):
        states, actions, rewards, next_states, dones = batch
        Q_s_1 = self.target_model(next_states).max(1)[0] # Compute the target network Q-function on s_{t+1}
        Q_target = rewards + (self.gamma * Q_s_1 * (1 - dones)) # Add the reward, to create the target
        Q_expected = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(Q_expected, Q_target)

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.soft_update()

        return loss.item()