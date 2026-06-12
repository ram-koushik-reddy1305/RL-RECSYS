import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        """
        DDPG Replay Buffer to store transition experiences.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of transitions and convert them to torch Tensors.
        
        Args:
            batch_size: Number of transitions to sample
            device: Target device (e.g., 'cpu' or 'cuda')
        Returns:
            states, actions, rewards, next_states, dones as torch Tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first for efficiency before loading to torch
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        dones_tensor = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
        
    def __len__(self) -> int:
        """
        Returns the current number of experiences stored.
        """
        return len(self.buffer)
