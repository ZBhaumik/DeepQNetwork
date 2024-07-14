import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        # Keep track of size to ensure circular buffer behaviour.
        self.max_size = buffer_size
        self.stored_experiences = 0
        self.count = 0

        # Initialize buffer elements
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.long)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.float)
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.max_size
        self.stored_experiences = min(self.max_size, self.stored_experiences + 1)

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self, batch_size):
        if (batch_size > self.stored_experiences):
            indices = np.linspace(0, self.stored_experiences, endpoint=False)
        else:
            indices = np.random.choice(self.stored_experiences, batch_size, replace=False)
        return (self.state[indices].to(self.get_device()),
                self.action[indices].to(self.get_device()),
                self.reward[indices].to(self.get_device()),
                self.next_state[indices].to(self.get_device()),
                self.done[indices].to(self.get_device()))