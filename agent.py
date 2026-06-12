import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        actor_lr: float = 1e-4, 
        critic_lr: float = 1e-3, 
        gamma: float = 0.9, 
        tau: float = 0.001, 
        buffer_capacity: int = 100000,
        device: str = None
    ):
        """
        DDPG Agent implementation for recommender systems.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector (item embedding size)
            actor_lr: Learning rate for the Actor network
            critic_lr: Learning rate for the Critic network
            gamma: Discount factor
            tau: Target network soft-update coefficient
            buffer_capacity: Capacity of replay buffer
            device: Device to use (e.g., 'cuda' or 'cpu'). Defaults to auto-detect.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # Copy original weights to targets
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Loss function
        self.critic_criterion = nn.MSELoss()
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """
        Select action (continuous item embedding) given the current state.
        
        Args:
            state: Array of shape (state_dim,)
            noise_std: Standard deviation of Gaussian noise to add for exploration (0.0 during evaluation)
        Returns:
            action: Array of shape (action_dim,) representing item embedding recommendation
        """
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)
            action_tensor = self.actor(state_tensor)                              # shape: (1, action_dim)
            action = action_tensor.cpu().numpy()[0]                               # shape: (action_dim,)
            
        # Add exploration noise
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=self.action_dim)
            action = action + noise
            
        return action
        
    def update(self, batch_size: int) -> tuple[float, float]:
        """
        Perform a single DDPG update step.
        
        Args:
            batch_size: Size of training batch
        Returns:
            critic_loss, actor_loss: The scalar loss values from this step (or 0.0, 0.0 if buffer too small)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
            
        self.actor.train()
        self.critic.train()
        
        # 1. Sample batch from Replay Buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, device=self.device)
        
        # 2. Update Critic
        with torch.no_grad():
            # Get target actions from target actor
            target_next_actions = self.target_actor(next_states)
            # Get target Q-values from target critic
            target_q_values = self.target_critic(next_states, target_next_actions)
            # Bellman equation: target_Q = reward + gamma * target_Q_next * (1 - done)
            target_q = rewards + self.gamma * target_q_values * (1.0 - dones)
            
        # Get current Q-values estimation
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = self.critic_criterion(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 3. Update Actor (Policy Gradient)
        # We maximize the expected return: actor_loss = -mean(Q(s, actor(s)))
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 4. Soft Update Targets
        self._soft_update(self.actor, self.target_actor, self.tau)
        self._soft_update(self.critic, self.target_critic, self.tau)
        
        return critic_loss.item(), actor_loss.item()
        
    def _soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """
        Soft update model parameters: target = tau * local + (1 - tau) * target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save(self, filepath_dir: str, prefix: str = ""):
        """
        Save Actor/Critic model weights.
        """
        os.makedirs(filepath_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(filepath_dir, f"{prefix}actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(filepath_dir, f"{prefix}critic.pth"))
        
    def load(self, filepath_dir: str, prefix: str = ""):
        """
        Load Actor/Critic model weights.
        """
        self.actor.load_state_dict(torch.load(os.path.join(filepath_dir, f"{prefix}actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(filepath_dir, f"{prefix}critic.pth"), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
