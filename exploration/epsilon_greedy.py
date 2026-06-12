import numpy as np

class EpsilonGreedyExplorer:
    def __init__(self, initial_epsilon: float = 1.0, decay_rate: float = 0.995, min_epsilon: float = 0.1):
        """
        Epsilon-Greedy Explorer for DDPG Recommender System.
        
        Args:
            initial_epsilon: Starting exploration probability
            decay_rate: Factor by which epsilon is multiplied after each episode
            min_epsilon: Minimum value of epsilon
        """
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
    def select_action(self, agent, state: np.ndarray, item_embeddings: np.ndarray, history: list[int]) -> np.ndarray:
        """
        Selects an item embedding action using epsilon-greedy strategy.
        
        Args:
            agent: The DDPGAgent instance
            state: State vector of shape (state_dim,)
            item_embeddings: Matrix of all item embeddings of shape (num_items, action_dim)
            history: List of item indices already recommended in the current session
        Returns:
            action: Selected action embedding of shape (action_dim,)
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # 1. Explore: Recommend a random available item
            num_items = item_embeddings.shape[0]
            # Exclude items already recommended in this session
            available_indices = [idx for idx in range(num_items) if idx not in history]
            
            if len(available_indices) == 0:
                random_idx = np.random.randint(0, num_items)
            else:
                random_idx = np.random.choice(available_indices)
                
            # Return the exact embedding of the chosen random item
            action = item_embeddings[random_idx]
        else:
            # 2. Exploit: Use Actor's output embedding directly
            action = agent.select_action(state, noise_std=0.0)
            
        return action
        
    def decay(self):
        """
        Decay epsilon at the end of each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
