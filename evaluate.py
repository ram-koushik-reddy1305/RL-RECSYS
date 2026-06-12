import os
import json
import numpy as np
import torch
from env import KuaiRecEnvironment
from agent import DDPGAgent

def calculate_ndcg(recommended_relevances, ideal_relevances, k):
    """
    Computes NDCG@K for a list of recommended items' relevances and ideal relevances.
    """
    rec_rel = recommended_relevances[:k]
    ideal_rel = ideal_relevances[:k]
    
    # Calculate DCG
    dcg = sum((2 ** r - 1) / np.log2(idx + 2) for idx, r in enumerate(rec_rel))
    
    # Calculate IDCG
    idcg = sum((2 ** r - 1) / np.log2(idx + 2) for idx, r in enumerate(ideal_rel))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def calculate_map(recommended_relevances, relevance_threshold, total_liked, k):
    """
    Computes Mean Average Precision (MAP) @ K for recommended item relevances.
    """
    rec_rel = recommended_relevances[:k]
    binary_relevances = (rec_rel > relevance_threshold).astype(int)
    
    ap = 0.0
    hits = 0.0
    for i, rel in enumerate(binary_relevances):
        if rel == 1:
            hits += 1.0
            ap += hits / (i + 1)
            
    denom = min(k, total_liked) if total_liked > 0 else k
    if denom == 0:
        return 0.0
    return ap / denom

def evaluate(
    checkpoint_dir: str = "./checkpoints", 
    num_test_users: int = 50, 
    k_list: list[int] = [1, 5, 10], 
    relevance_threshold: float = -0.5,  # Since ratings are normalized to [-1, 1] with clip=2.0, > -0.5 represents watch_ratio > 0.5
    experiment_name: str = "drrmax_baseline"
):
    """
    Evaluates the trained DDPG agent on a set of test users.
    Calculates Precision@K, Recall@K, NDCG@K, MAP@K, and Average Cumulative Reward.
    """
    print(f"Initializing Evaluation for experiment: {experiment_name}...")
    env = KuaiRecEnvironment(
        csv_file_path='kuairec_final.csv',
        pmf_user_path='pmf1_user_embeddings.npy',
        pmf_item_path='pmf1_item_embeddings.npy',
        latent_dim=200,
        max_steps=10
    )
    
    agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.latent_dim)
    
    # Load weights from checkpoint
    try:
        agent.load(checkpoint_dir, prefix=f"{experiment_name}_")
        print(f"Loaded trained models from {checkpoint_dir}/ with prefix {experiment_name}_")
    except Exception as e:
        print(f"Warning: Could not load trained models from {checkpoint_dir}. Reason: {e}")
        print("Proceeding with randomly initialized agent for testing.")

    # Dictionary to store metrics for each K
    metrics = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in k_list}
    rewards_list = []
    
    # Evaluate over test users
    for user_idx in range(num_test_users):
        # Sample test user from the environment
        user_id = user_idx % env.num_users
        state = env.reset(user_id)
        
        session_rewards = []
        user_precision = {k: [] for k in k_list}
        user_recall = {k: [] for k in k_list}
        user_ndcg = {k: [] for k in k_list}
        user_map = {k: [] for k in k_list}
        
        done = False
        while not done:
            # Action embedding output by Actor (no noise during evaluation)
            action = agent.select_action(state, noise_std=0.0)
            
            # 1. Simulate the recommendation step
            next_state, reward, done, info = env.step(action)
            session_rewards.append(reward)
            
            # 2. Compute retrieval metrics at this step
            max_k = max(k_list)
            scores = np.dot(env.item_embeddings, action)
            
            # Mask out items recommended in previous steps of this session to avoid duplicates
            history_before_step = env.history[:-1]
            for item_idx in history_before_step:
                scores[item_idx] = -np.inf
                
            top_k_indices = np.argsort(scores)[::-1][:max_k]
            
            # Fetch true ratings of these recommended items from the ratings matrix
            user_ratings = env.ratings_matrix[env.current_user_idx]
            actual_relevances = user_ratings[top_k_indices]
            
            # Calculate total liked items in the entire catalog for this user (ground truth profile)
            total_liked_items = np.sum(user_ratings > relevance_threshold)
            
            # Compute metrics for each K
            for k in k_list:
                rec_relevances = actual_relevances[:k]
                liked_in_rec = np.sum(rec_relevances > relevance_threshold)
                
                # Precision@K
                precision = liked_in_rec / k
                user_precision[k].append(precision)
                
                # Recall@K
                recall = liked_in_rec / total_liked_items if total_liked_items > 0 else 0.0
                user_recall[k].append(recall)
                
                # NDCG@K
                ideal_relevances = sorted(actual_relevances, reverse=True)
                ndcg = calculate_ndcg(rec_relevances, ideal_relevances, k)
                user_ndcg[k].append(ndcg)
                
                # MAP@K
                map_score = calculate_map(actual_relevances, relevance_threshold, total_liked_items, k)
                user_map[k].append(map_score)
                
            state = next_state
            
        rewards_list.append(sum(session_rewards))
        
        for k in k_list:
            metrics[k]["precision"].append(np.mean(user_precision[k]))
            metrics[k]["recall"].append(np.mean(user_recall[k]))
            metrics[k]["ndcg"].append(np.mean(user_ndcg[k]))
            metrics[k]["map"].append(np.mean(user_map[k]))
            
    # Compile and display final results
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS: {experiment_name}")
    print(f"Average Cumulative Reward: {np.mean(rewards_list):.4f}")
    print("-"*40)
    for k in k_list:
        print(f"K = {k}:")
        print(f"  Precision@{k} : {np.mean(metrics[k]['precision']):.4f}")
        print(f"  Recall@{k}    : {np.mean(metrics[k]['recall']):.4f}")
        print(f"  NDCG@{k}      : {np.mean(metrics[k]['ndcg']):.4f}")
        print(f"  MAP@{k}       : {np.mean(metrics[k]['map']):.4f}")
    print("="*40 + "\n")
    
    # Save the final scores to results directory
    metrics_data = {
        "avg_cumulative_reward": float(np.mean(rewards_list)),
        "precision": {str(k): float(np.mean(metrics[k]["precision"])) for k in k_list},
        "recall": {str(k): float(np.mean(metrics[k]["recall"])) for k in k_list},
        "ndcg": {str(k): float(np.mean(metrics[k]["ndcg"])) for k in k_list},
        "map": {str(k): float(np.mean(metrics[k]["map"])) for k in k_list}
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/{experiment_name}_evaluation.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved evaluation metrics to results/{experiment_name}_evaluation.json")
    
    return metrics, np.mean(rewards_list)

if __name__ == "__main__":
    evaluate(experiment_name="drrmax_baseline")
