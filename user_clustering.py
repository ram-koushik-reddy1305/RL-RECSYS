import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def generate_user_clusters(
    pmf_user_path='pmf1_user_embeddings.npy',
    train_csv_path='kuairec_train_split.csv',
    num_clusters=10,
    top_m=50,
    output_dir='results'
):
    print(f"Loading user embeddings from {pmf_user_path}...")
    user_embeddings = np.load(pmf_user_path)  # Shape: (num_users, 200)
    num_users = user_embeddings.shape[0]
    
    print(f"Clustering {num_users} users into {num_clusters} clusters using KMeans...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(user_embeddings)
    centroids = kmeans.cluster_centers_  # Shape: (num_clusters, 200)
    
    # Create mapping of User Index -> Cluster ID
    user_to_cluster = {int(u_idx): int(c_id) for u_idx, c_id in enumerate(cluster_labels)}
    
    print(f"Loading training data from {train_csv_path} to calculate cluster popular items...")
    df = pd.read_csv(train_csv_path)
    
    # Sort chronologically by user and timestamp to ensure maps match pmf.py
    if 'timestamp' in df.columns:
        df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        
    # Recreate the exact ID maps from env.py to align user/item indexing
    user_map = {id: idx for idx, id in enumerate(df['user_id'].unique())}
    item_map = {id: idx for idx, id in enumerate(df['video_id'].unique())}
    
    # Add mapped user and item indices to the dataframe
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['video_id'].map(item_map)
    
    # Map user cluster IDs to interactions
    df['cluster_idx'] = df['user_idx'].map(user_to_cluster)
    
    # Calculate the top M popular items for each cluster (by watch ratio/rating frequency)
    cluster_popular_items = {}
    print("Calculating popular items for each cluster...")
    for c_id in range(num_clusters):
        cluster_df = df[df['cluster_idx'] == c_id]
        if cluster_df.empty:
            cluster_popular_items[c_id] = []
            continue
            
        # Group by item index and compute the mean watch ratio/popularity count
        item_stats = cluster_df.groupby('item_idx').agg(
            popularity=('item_idx', 'count'),
            mean_watch_ratio=('watch_ratio', 'mean')
        ).reset_index()
        
        # Sort by popularity count, then by watch ratio
        top_items = item_stats.sort_values(by=['popularity', 'mean_watch_ratio'], ascending=False).head(top_m)
        cluster_popular_items[c_id] = [int(i) for i in top_items['item_idx'].values]
        
    # Compute Euclidean distance between cluster centroids to find "nearby" clusters
    print("Computing nearby cluster relations...")
    nearby_clusters = {}
    for c_id in range(num_clusters):
        centroid = centroids[c_id]
        distances = []
        for other_id in range(num_clusters):
            if other_id == c_id:
                continue
            dist = np.linalg.norm(centroid - centroids[other_id])
            distances.append((other_id, dist))
        
        # Sort by distance and retrieve the closest clusters (e.g. top 3 closest clusters)
        sorted_distances = sorted(distances, key=lambda x: x[1])
        nearby_clusters[c_id] = [int(x[0]) for x in sorted_distances[:3]]
        
    # Create results folder if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the lookup tables as JSON files
    with open(os.path.join(output_dir, 'user_to_cluster.json'), 'w') as f:
        json.dump(user_to_cluster, f)
        
    with open(os.path.join(output_dir, 'cluster_popular_items.json'), 'w') as f:
        json.dump(cluster_popular_items, f)
        
    with open(os.path.join(output_dir, 'nearby_clusters.json'), 'w') as f:
        json.dump(nearby_clusters, f)
        
    print(f"Offline clustering completed. Mappings saved to: {output_dir}/")

if __name__ == "__main__":
    generate_user_clusters(num_clusters=10, top_m=50)
