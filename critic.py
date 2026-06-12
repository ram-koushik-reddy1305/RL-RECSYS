import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim1: int = 256, hidden_dim2: int = 128):
        """
        Critic Network for DDPG Recommender System.
        Evaluates the Q-value Q(s, a) of a given state and action.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector (item embedding dimension)
            hidden_dim1: Size of the first hidden layer
            hidden_dim2: Size of the second hidden layer
        """
        super(Critic, self).__init__()
        # Concatenate state and action at the input layer
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Maps (state, action) -> Q-value.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)
        Returns:
            q_value: Tensor of shape (batch_size, 1)
        """
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
