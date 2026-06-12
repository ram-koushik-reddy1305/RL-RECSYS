import numpy as np
import pandas as pd
import torch
from state_generation import StateGenerationModule

class KuaiRecEnvironment:
    def __init__(self, csv_file_path='kuairec_final.csv', pmf_user_path='pmf1_user_embeddings.npy', pmf_item_path='pmf1_item_embeddings.npy', latent_dim=200, max_steps=10):
        """
        True simulator environment for KuaiRec.
        Uses the dense small matrix to look up real user feedback (watch ratio) for recommendations.
        """
        self.max_steps = max_steps
        self.latent_dim = latent_dim
        self.state_dim = 3 * latent_dim
        
        print("Env: Loading KuaiRec dataset and constructing ratings lookup table...")
        df = pd.read_csv(csv_file_path)
        
        # Sort chronologically by user and timestamp to ensure maps match pmf.py
        if 'timestamp' in df.columns:
            df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
            
        # Recreate the exact ID maps from pmf.py
        self.user_map = {id: idx for idx, id in enumerate(df['user_id'].unique())}
        self.item_map = {id: idx for idx, id in enumerate(df['video_id'].unique())}
        
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        # Map ratings (clip watch ratio to 2.0 and normalize to [-1, 1])
        df['rating'] = df['watch_ratio'].clip(upper=2.0)
        min_r, max_r = 0.0, 2.0
        df['normalized_rating'] = 2.0 * (df['rating'] - min_r) / (max_r - min_r) - 1.0
        
        # Build dense lookup ratings matrix of shape (num_users, num_items)
        # Initialize to -1.0 (representing un-interacted or minimum watch ratio)
        self.ratings_matrix = np.ones((self.num_users, self.num_items)) * -1.0
        u_indices = df['user_id'].map(self.user_map).values
        i_indices = df['video_id'].map(self.item_map).values
        ratings = df['normalized_rating'].values
        self.ratings_matrix[u_indices, i_indices] = ratings
        
        # Initialize State Generation Module (frozen coordinates)
        self.state_generator = StateGenerationModule(
            pmf_user_path=pmf_user_path,
            pmf_item_path=pmf_item_path,
            latent_dim=latent_dim
        )
        
        # Store item embeddings for candidate selection
        self.item_embeddings = self.state_generator.item_embeddings.weight.detach().cpu().numpy()
        
        self.current_user_idx = None
        self.history = []
        self.current_step = 0
        
    def reset(self, user_idx=None) -> np.ndarray:
        """
        Reset environment for a session.
        """
        if user_idx is None:
            self.current_user_idx = np.random.randint(0, self.num_users)
        else:
            self.current_user_idx = user_idx
            
        # Find all items this user has interacted with
        user_ratings = self.ratings_matrix[self.current_user_idx]
        interacted_items = np.where(user_ratings != 0)[0]
        
        if len(interacted_items) >= 10:
            self.history = list(interacted_items[:10])
        else:
            # Fallback if user has too few interactions (should not happen on KuaiRec filtered data)
            self.history = list(interacted_items) + list(np.random.choice(self.num_items, 10 - len(interacted_items), replace=False))
            
        self.current_step = 0
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        """
        Generate state vector via Person A's StateGenerationModule.
        """
        user_tensor = torch.tensor([self.current_user_idx], dtype=torch.long)
        history_tensor = torch.tensor([self.history], dtype=torch.long)
        
        with torch.no_grad():
            state_tensor = self.state_generator(user_tensor, history_tensor)
            
        return state_tensor.cpu().numpy()[0]
        
    def step(self, action_emb: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Recommends the closest candidate item and retrieves user feedback.
        """
        # Cosine similarity matching to item catalog
        scores = np.dot(self.item_embeddings, action_emb)
        
        # Mask already recommended items
        for item_idx in self.history:
            scores[item_idx] = -np.inf
            
        recommended_item_idx = int(np.argmax(scores))
        self.history.append(recommended_item_idx)
        self.history.pop(0)  # Maintain rolling window of 10 items
        
        # Get true user reward
        reward = self.ratings_matrix[self.current_user_idx, recommended_item_idx]
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state()
        info = {
            'recommended_item_id': recommended_item_idx,
            'user_idx': self.current_user_idx
        }
        
        return next_state, reward, done, info
