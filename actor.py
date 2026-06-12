import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim1: int = 256, hidden_dim2: int = 128):
        """
        Actor Network for DDPG Recommender System.
        Outputs a continuous action vector (item embedding).
        
        Args:
            state_dim: Dimension of state vector (e.g., user + history + interaction)
            action_dim: Dimension of action vector (item embedding dimension)
            hidden_dim1: Size of the first hidden layer
            hidden_dim2: Size of the second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Maps state -> continuous action embedding.
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            action: Continuous embedding tensor of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Continuous action representing the desired item embedding.
        # No Tanh is applied at the output since item embeddings from PMF are typically real-valued.
        action = self.fc3(x)
        return action
