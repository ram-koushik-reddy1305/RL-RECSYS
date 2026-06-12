import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prevent memory fragmentation on Kaggle GPU
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load data
df = pd.read_csv('/kaggle/input/datasets/dayanithaam/small-matrix/kuairec_final.csv')

# Ensure chronological consistency across user streams before slicing splits
if 'timestamp' in df.columns:
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

# Calculate normalized pseudo-ratings between -1 and 1
df['rating'] = df['watch_ratio']
max_r, min_r = df['rating'].max(), df['rating'].min()
df['normalized_rating'] = 2.0 * (df['rating'] - min_r) / (max_r - min_r) - 1.0

# Map IDs to continuous indices for embedding lookups (computed globally so dimensions match)
user_map = {id: idx for idx, id in enumerate(df['user_id'].unique())}
item_map = {id: idx for idx, id in enumerate(df['video_id'].unique())}

num_users = len(user_map)
num_items = len(item_map)

df['u_idx'] = df['user_id'].map(user_map)
df['i_idx'] = df['video_id'].map(item_map)

# ─── DATA SPLITTING WITHOUT LEAKAGE ──────────────────────────────────
# Perform chronological 80% train and 20% test split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"Total Rows: {len(df)} | Train Rows: {len(train_df)} | Test Rows: {len(test_df)}")

# Map only the training slice down to PyTorch execution tensors
users_t = torch.tensor(train_df['u_idx'].values, dtype=torch.long)
items_t = torch.tensor(train_df['i_idx'].values, dtype=torch.long)
ratings_t = torch.tensor(train_df['normalized_rating'].values, dtype=torch.float)

# ─── PYTORCH MODEL ───────────────────────────────────────────────────
class PMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=200):
        super(PMF, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, users, items):
        u_emb = self.user_embeddings(users)
        i_emb = self.item_embeddings(items)
        return (u_emb * i_emb).sum(dim=1)

# 3. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 200 # Matching specification [cite: 481]
pmf_model = PMF(num_users, num_items, latent_dim).to(device)

criterion = nn.MSELoss()
# Using optim.Adam directly to clear up namespace optimizer calls
opt = optim.Adam(pmf_model.parameters(), lr=0.01, weight_decay=1e-4) 

dataset = TensorDataset(users_t, items_t, ratings_t)
loader = DataLoader(dataset, batch_size=4096, shuffle=True)

best_loss = float('inf')

# 4. Run the PMF Optimization Loop on Train Data Only
print("Training true PMF baseline lookup matrices on chronological train split...")
pmf_model.train()
for epoch in range(20):
    total_loss = 0
    for users_b, items_b, ratings_b in loader:
        users_b, items_b, ratings_b = users_b.to(device), items_b.to(device), ratings_b.to(device)
        
        opt.zero_grad()
        predictions = pmf_model(users_b, items_b)
        loss = criterion(predictions, ratings_b)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/20 | Avg Train Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(pmf_model.state_dict(), 'pmf_best.pth')

print(f"Best Training Loss reached: {best_loss:.4f}")

# Load best weights before extracting
pmf_model.load_state_dict(torch.load('pmf_best.pth'))
pmf_model.eval()

# Export finalized baseline mappings
U_matrix = pmf_model.user_embeddings.weight.detach().cpu().numpy()
I_matrix = pmf_model.item_embeddings.weight.detach().cpu().numpy()

np.save('pmf1_user_embeddings.npy', U_matrix)
np.save('pmf1_item_embeddings.npy', I_matrix)

# Optional helper check to view test dataset partitions from the notebook cells downstream
test_df.to_csv('kuairec_test_split.csv', index=False)

print(f"Saved clean PMF lookup keys. U shape: {U_matrix.shape}, I shape: {I_matrix.shape}")