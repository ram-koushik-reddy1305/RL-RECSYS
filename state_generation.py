# prepare_rl_environment.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────
# 1. THE SEQUENTIAL TRAJECTORY DATASET DEFINITION
# ──────────────────────────────────────────────────────────────────────
class KuaiRecSequentialRLDataset(Dataset):
    def __init__(self, csv_file_path, window_size=10):
        """
        csv_file_path: Path to 'kuairec_train_split.csv' or 'kuairec_test_split.csv'
        window_size: The history window size N (e.g., 10)
        """
        self.samples = []
        self.window_size = window_size
        
        print(f"Reading dataset from: {csv_file_path}...")
        dataframe = pd.read_csv(csv_file_path)
        
        print("Processing records into rolling interaction histories...")
        # Group by user to keep individual chronological timelines intact
        for u_idx, group in dataframe.groupby('u_idx'):
            # If a user's chronological fragment is too short for the window, skip them safely
            if len(group) >= window_size + 1:
                items = group['i_idx'].values.tolist()
                ratings = group['normalized_rating'].values.tolist()
                
                # Create rolling history windows
                # 'history' tracks steps [i-N to i], and the item at index 'i' acts as the target next interaction
                for i in range(window_size, len(items)):
                    history = items[i - window_size:i]
                    target_item = items[i]
                    reward = ratings[i]  # Watch ratio feedback metric acting as reward
                    
                    self.samples.append((u_idx, history, target_item, reward))
                    
        print(f"Extraction complete. Formed {len(self.samples)} sequence trajectories.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u_idx, history, target_item, reward = self.samples[idx]
        
        # Return as PyTorch Tensors ready for RL execution
        return (
            torch.tensor(u_idx, dtype=torch.long),
            torch.tensor(history, dtype=torch.long),
            torch.tensor(target_item, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float)
        )


# ──────────────────────────────────────────────────────────────────────
# 2. THE STATE GENERATION MODULE DEFINITION
# ──────────────────────────────────────────────────────────────────────
class StateGenerationModule(nn.Module):
    def __init__(self, pmf_user_path='pmf1_user_embeddings.npy', pmf_item_path='pmf1_item_embeddings.npy', latent_dim=200):
        super(StateGenerationModule, self).__init__()
        
        # Load the pre-computed baseline lookup arrays generated from your PMF step
        u_init = np.load(pmf_user_path)
        i_init = np.load(pmf_item_path)
        
        # Wrap them inside PyTorch Embedding layers and FREEZE them.
        # This keeps your coordinate system stable across the DRL training phase
        self.user_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(u_init), freeze=True)
        self.item_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(i_init), freeze=True)
        
        self.latent_dim = latent_dim

    def forward(self, user_id, history_item_ids):
        """
        user_id: Tensor of shape (batch_size,)
        history_item_ids: Tensor of shape (batch_size, N) where N is your sliding window size
        """
        # Look up long-term preference vector 'u' -> Shape: (batch_size, k)
        u = self.user_embeddings(user_id)
        
        # Look up history item vectors -> Shape: (batch_size, N, k)
        history_items = self.item_embeddings(history_item_ids)
        
        # Equation (2): Max Pooling along the sequence dimension (dim=1) to isolate short-term interest
        short_term, _ = torch.max(history_items, dim=1)
        
        # Element-wise product (\otimes) mapping user profile onto recent habits -> Shape: (batch_size, k)
        interaction = u * short_term
        
        # Equation (1): Concatenate and flatten into a unified state matrix -> Shape: (batch_size, 3k)
        state = torch.cat([u, interaction, short_term], dim=1)
        
        return state


# ──────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT GENERATION & HAND-OFF SERIALIZATION EXECUTOR
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WINDOW_SIZE_N = 10
    
    # Paths to your input files
    TRAIN_CSV = 'kuairec_train_split.csv'
    TEST_CSV = 'kuairec_test_split.csv'
    
    print("=== STARTING ENVIRONMENT GENERATION PROCESSING ===")
    
    # 1. Process the CSV rows into memory-mapped sequential datasets
    train_dataset = KuaiRecSequentialRLDataset(TRAIN_CSV, window_size=WINDOW_SIZE_N)
    test_dataset = KuaiRecSequentialRLDataset(TEST_CSV, window_size=WINDOW_SIZE_N)
    
    # 2. Serialize and save the processed dataset classes directly to disk
    # This stores the internal self.samples lists so your teammate can load them instantly
    print("\nSaving processed dataset objects for your teammate...")
    torch.save(train_dataset, 'processed_train_rl_dataset.pt')
    torch.save(test_dataset, 'processed_test_rl_dataset.pt')
    print("Saved 'processed_train_rl_dataset.pt' and 'processed_test_rl_dataset.pt' successfully.")
    
    # 3. Initialize the State Generation Module for validation
    print("\nInitializing State Generation Module verification...")
    state_generator = StateGenerationModule(
        pmf_user_path='pmf1_user_embeddings.npy',
        pmf_item_path='pmf1_item_embeddings.npy',
        latent_dim=200
    )
    
    # 4. Create a temporary data loader to run a trial state check
    verify_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_users, batch_histories, batch_targets, batch_rewards in verify_loader:
        # Pass data through your module to make sure shapes match up with DRR specifications
        sample_states = state_generator(batch_users, batch_histories)
        
        print("\n--- Verification Sample Shapes Passed ---")
        print(f"Batch User Indices Shape:      {batch_users.shape}")
        print(f"Batch History Matrices Shape:  {batch_histories.shape}")
        print(f"Resulting State Matrix Shape:  {sample_states.shape} (Expected: [4, 600])")
        print("==================================================")
        break