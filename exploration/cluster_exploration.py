import os
import json
import numpy as np
import torch

class ClusterExplorationExplorer:
    def __init__(
        self,
        item_embeddings: np.ndarray,
        initial_epsilon: float = 1.0,
        decay_rate: float = 0.995,
        min_epsilon: float = 0.1,
        mappings_dir: str = "results"
    ):
        """
        Nearby-Cluster Exploration (UCE) Explorer.
        
        Args:
            item_embeddings: Matrix of item embeddings (num_items, action_dim)
            initial_epsilon: Starting exploration probability
            decay_rate: Epsilon multiplier per episode
            min_epsilon: Lower limit for epsilon
            mappings_dir: Directory containing user clustering JSON files
        """
        self.item_embeddings = item_embeddings
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
        # Load offline clustering mappings
        print("UCE: Loading offline clustering mappings...")
        user_to_cluster_path = os.path.join(mappings_dir, 'user_to_cluster.json')
        cluster_popular_items_path = os.path.join(mappings_dir, 'cluster_popular_items.json')
        nearby_clusters_path = os.path.join(mappings_dir, 'nearby_clusters.json')
        
        if not (os.path.exists(user_to_cluster_path) and 
                os.path.exists(cluster_popular_items_path) and 
                os.path.exists(nearby_clusters_path)):
            raise FileNotFoundError(
                f"Required clustering maps not found in {mappings_dir}/. "
                "Please run user_clustering.py first."
            )
            
        with open(user_to_cluster_path, 'r') as f:
            # Convert keys to int (JSON keys are parsed as strings)
            self.user_to_cluster = {int(k): int(v) for k, v in json.load(f).items()}
            
        with open(cluster_popular_items_path, 'r') as f:
            self.cluster_popular_items = {int(k): list(v) for k, v in json.load(f).items()}
            
        with open(nearby_clusters_path, 'r') as f:
            self.nearby_clusters = {int(k): list(v) for k, v in json.load(f).items()}

    def select_action(self, agent, state: np.ndarray, user_idx: int, history: list[int]) -> np.ndarray:
        """
        Select action (item embedding) using Nearby-Cluster Exploration strategy.
        
        Args:
            agent: The DDPGAgent instance
            state: State vector of shape (state_dim,)
            user_idx: Index of current user
            history: List of item indices already recommended in active session
        Returns:
            action: Action embedding vector of shape (action_dim,)
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # 1. Explore: Select a popular item from nearby user clusters
            user_cluster = self.user_to_cluster.get(user_idx)
            if user_cluster is None:
                # Fallback to pure random exploration if user cluster is missing
                available_indices = [idx for idx in range(self.item_embeddings.shape[0]) if idx not in history]
                explore_idx = np.random.choice(available_indices) if len(available_indices) > 0 else np.random.randint(0, self.item_embeddings.shape[0])
                return self.item_embeddings[explore_idx]
                
            # Gather popular item indices from nearby clusters
            nearby_ids = self.nearby_clusters.get(user_cluster, [])
            candidate_items = []
            for n_c_id in nearby_ids:
                candidate_items.extend(self.cluster_popular_items.get(n_c_id, []))
                
            # Remove duplicates and items already recommended in this active session
            candidate_items = list(set(candidate_items))
            available_candidates = [idx for idx in candidate_items if idx not in history]
            
            if len(available_candidates) > 0:
                # Recommending a popular item from nearby clusters
                explore_idx = np.random.choice(available_candidates)
            else:
                # Fallback to catalog random if all nearby candidates are exhausted
                available_indices = [idx for idx in range(self.item_embeddings.shape[0]) if idx not in history]
                explore_idx = np.random.choice(available_indices) if len(available_indices) > 0 else np.random.randint(0, self.item_embeddings.shape[0])
                
            return self.item_embeddings[explore_idx]
        else:
            # 2. Exploit: Recommend Actor's optimal item representation directly (no noise)
            return agent.select_action(state, noise_std=0.0)

    def decay(self):
        """
        Decay epsilon at the end of each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
