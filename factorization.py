import pandas as pd
import numpy as np
from sklearn.decomposition import NMF # Or use a dedicated PMF library / custom PyTorch PMF loop

# 1. Load the core sequence interactions
# Adjust column names based on your exact KuaiRec file layout (e.g., small_matrix.csv)
df = pd.read_csv('kuairec_final.csv') 

# 2. Define a pseudo-rating value to serve as the structural anchor
# Example: Using watch ratio and explicit likes to create a custom target rating scale
df['pseudo_rating'] = df['watch_ratio'] 

# 3. Pivot into a sparse matrix form (Users x Items)
interaction_matrix_df = df.pivot(index='user_id', columns='video_id', values='pseudo_rating').fillna(0)
X = interaction_matrix_df.values

# 4. Extract constant latent dimensions (k = 200 as specified in paper setup)
k = 200
model = NMF(n_components=k, init='random', random_state=42, max_iter=200)
U_weights = model.fit_transform(X)       # User Latent Space Matrix (U)
I_components = model.components_         # Item Latent Space Matrix (I)

# Save matrices down as numpy or torch checkpoints to load directly into your RL pipeline
np.save('pmf_user_embeddings.npy', U_weights)
np.save('pmf_item_embeddings.npy', I_components.T)

print(f"Static embedding profiles initialized. User Shape: {U_weights.shape}, Item Shape: {I_components.T.shape}")