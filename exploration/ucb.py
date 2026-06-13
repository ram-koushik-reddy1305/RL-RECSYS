import numpy as np
import torch

class UCBExplorer:
    def __init__(self, num_items: int, c: float = 1.0):
        """
        UCB Explorer for DDPG Recommender System.
        
        Args:
            num_items: Total number of items in the environment.
            c: Exploration bonus coefficient.
        """
        self.c = c
        self.num_items = num_items
        self.visit_counts = np.zeros(num_items, dtype=np.int64)
        self.total_steps = 0
        
    def select_action(self, agent, state: np.ndarray, item_embeddings: np.ndarray, history: list[int]) -> np.ndarray:
        """
        Selects an item embedding action using UCB strategy.
        
        Args:
            agent: The DDPGAgent instance
            state: State vector of shape (state_dim,)
            item_embeddings: Matrix of all item embeddings of shape (num_items, action_dim)
            history: List of item indices already recommended in the current session
        Returns:
            action: Selected action embedding of shape (action_dim,)
        """
        num_candidates = item_embeddings.shape[0]
        # Exclude items already recommended in this session
        available_indices = [idx for idx in range(num_candidates) if idx not in history]
        
        if len(available_indices) == 0:
            best_idx = np.random.randint(0, num_candidates)
            self.visit_counts[best_idx] += 1
            self.total_steps += 1
            return item_embeddings[best_idx]
            
        available_embeddings = item_embeddings[available_indices]
        
        # Evaluate Q-values of candidate items using the Critic network.
        # We build a batch for the state-action input of the Critic.
        agent.critic.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)  # shape: (1, state_dim)
            # Repeat state across all candidate items
            state_batch = state_tensor.repeat(len(available_indices), 1)  # shape: (len(available_indices), state_dim)
            action_batch = torch.FloatTensor(available_embeddings).to(agent.device)  # shape: (len(available_indices), action_dim)
            
            # Predict Q-values: shape (len(available_indices), 1)
            q_values = agent.critic(state_batch, action_batch).cpu().numpy().flatten()
            
        # Retrieve visitation counts for available items
        visit_counts_avail = self.visit_counts[available_indices]
        
        # Compute UCB scores: Q(s, a) + c * sqrt(ln(total_steps + 1) / (visit_counts + 1))
        # Adding 1 in the logs and denominator ensures numerical stability
        ucb_bonus = self.c * np.sqrt(np.log(self.total_steps + 1) / (visit_counts_avail + 1))
        scores = q_values + ucb_bonus
        
        # Find index with highest UCB score
        best_avail_idx = np.argmax(scores)
        best_global_idx = available_indices[best_avail_idx]
        
        # Update statistics
        self.visit_counts[best_global_idx] += 1
        self.total_steps += 1
        
        return item_embeddings[best_global_idx]
