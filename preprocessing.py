import pandas as pd

# Load your sequential log (adjust filename to your KuaiRec file, e.g., 'small_matrix.csv')
df = pd.read_csv('C:/RL DRR_MAX/RL-RECSYS/KuaiRec 2.0/data/small_matrix.csv')

# Step 1: Sort everything chronologically by user and timestamp
df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

# Step 2: Count how many interactions each user has
user_counts = df['user_id'].value_counts()

# Step 3: Identify users who meet your minimum history requirement (N = 10)
valid_users = user_counts[user_counts >= 10].index

# Step 4: Filter the dataframe to keep only those valid users
filtered_df = df[df['user_id'].isin(valid_users)].reset_index(drop=True)

print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
print(f"Dropped {len(user_counts) - len(valid_users)} users with fewer than 10 interactions.")

# Save this clean sequential dataset
filtered_df.to_csv('kuairec_final.csv', index=False)