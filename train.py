import os
import numpy as np
import pandas as pd
from env import KuaiRecEnvironment
from agent import DDPGAgent

def train(
    num_episodes: int = 200, 
    batch_size: int = 128, 
    noise_std: float = 0.1, 
    actor_lr: float = 1e-4, 
    critic_lr: float = 1e-3,
    save_dir: str = "./checkpoints",
    experiment_name: str = "drrmax_baseline"
):
    """
    DDPG Training Loop for Recommender System integrated with PMF embeddings and State Generation Module.
    """
    print("Initializing Environment and Agent...")
    env = KuaiRecEnvironment(
        csv_file_path='kuairec_train_split.csv',
        pmf_user_path='pmf1_user_embeddings.npy',
        pmf_item_path='pmf1_item_embeddings.npy',
        latent_dim=200,
        max_steps=10
    )
    
    # State dimension is 600, action dimension is 200
    agent = DDPGAgent(
        state_dim=env.state_dim, 
        action_dim=env.latent_dim, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr
    )
    
    rewards_history = []
    actor_losses = []
    critic_losses = []
    
    print(f"Starting DDPG Training for {num_episodes} episodes...")
    for episode in range(1, num_episodes + 1):
        # Sample a random user from the dataset to reset env
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        ep_actor_losses = []
        ep_critic_losses = []
        
        while not done:
            # Select action (desired item embedding) with exploration noise
            action = agent.select_action(state, noise_std=noise_std)
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            
            # Save experience to buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent parameters
            critic_loss, actor_loss = agent.update(batch_size)
            if critic_loss > 0 or actor_loss > 0:
                ep_critic_losses.append(critic_loss)
                ep_actor_losses.append(actor_loss)
                
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        
        # Calculate mean losses
        mean_critic_loss = np.mean(ep_critic_losses) if len(ep_critic_losses) > 0 else 0.0
        mean_actor_loss = np.mean(ep_actor_losses) if len(ep_actor_losses) > 0 else 0.0
        
        actor_losses.append(mean_actor_loss)
        critic_losses.append(mean_critic_loss)
        
        # Logging progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Avg Reward (last 10): {avg_reward:.4f} | "
                f"Buffer Size: {len(agent.replay_buffer)} | "
                f"Critic Loss: {mean_critic_loss:.4f} | "
                f"Actor Loss: {mean_actor_loss:.4f}"
            )
            
    # Save trained networks
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir, prefix=f"{experiment_name}_")
    print(f"Training completed. Models saved to {save_dir}/ with prefix {experiment_name}_")
    
    # Save training metrics to results folder
    os.makedirs("results", exist_ok=True)
    df_logs = pd.DataFrame({
        'Episode': range(1, num_episodes + 1),
        'Average_Reward': rewards_history,
        'Critic_Loss': critic_losses,
        'Actor_Loss': actor_losses
    })
    df_logs.to_csv(f'results/{experiment_name}_training.csv', index=False)
    print(f"Saved training logs to results/{experiment_name}_training.csv")
    
    return rewards_history, actor_losses, critic_losses

if __name__ == "__main__":
    train(num_episodes=200, experiment_name="drrmax_baseline")
