import numpy as np
import torch
from environment import NetworkSlicingEnv
from agent import IDDPGAgent, MAReplayBuffer, MADDPG
from optimizer import MasterOptimizer
import matplotlib.pyplot as plt

def main():
    # --- Hyperparameters ---
    NUM_EPISODES = 10 # Reduced for faster execution in this environment
    EPISODE_LENGTH = 100 # Reduced for faster execution
    START_TIMESTEPS = 500 # Ensure learning starts within the shorter timeframe
    BATCH_SIZE = 100
    ADMM_STEP_SIZE = 0.5 # Corresponds to rho in the paper/environment

    # --- Initialization ---
    env = NetworkSlicingEnv(rho=ADMM_STEP_SIZE)
    num_slices = env.num_slices

    # Initialize individual agents (actors)
    agents = []
    state_dims = []
    action_dims = []
    for i in range(num_slices):
        slice_name = f"slice_{i}"
        state_dim = env.observation_space[slice_name].shape[0]
        action_dim = env.action_space[slice_name].shape[0]
        max_action = env.action_space[slice_name].high[0]

        # We use the IDDPGAgent class which holds the actor and its optimizer
        agent = IDDPGAgent(state_dim, action_dim, max_action)
        agents.append(agent)
        state_dims.append(state_dim)
        action_dims.append(action_dim)
        print(f"Initialized agent for {slice_name}: state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")

    # Initialize the MADDPG Coordinator
    maddpg_coordinator = MADDPG(agents, sum(state_dims), sum(action_dims))
    replay_buffer = MAReplayBuffer(num_slices, state_dims, action_dims)
    print(f"\nInitialized MADDPG Coordinator and Multi-Agent Replay Buffer.")

    # Initialize the Master Optimizer
    master_optimizer = MasterOptimizer(num_slices, env.total_resources)
    print(f"Initialized Master Optimizer.")

    # Logging
    episode_rewards_log = []
    total_timesteps = 0

    # --- Main Training Loop ---
    print("\n--- Starting Training with MADDPG + ADMM Coordination ---")
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0
        z_vars = np.zeros(num_slices)
        y_vars = np.zeros(num_slices)

        for t in range(EPISODE_LENGTH):
            # 1. Action selection from each agent's actor
            if total_timesteps < START_TIMESTEPS:
                actions = {f'slice_{i}': env.action_space[f'slice_{i}'].sample() for i in range(num_slices)}
            else:
                actions = maddpg_coordinator.get_actions(obs)

            # 2. Master Problem (ADMM)
            x_totals = np.array([np.sum(actions[f"slice_{i}"]) for i in range(num_slices)])
            z_vars = master_optimizer.solve(x_totals, y_vars)
            if z_vars is None: z_vars = np.zeros(num_slices) # Fallback

            # 3. Dual Update (ADMM)
            y_vars += ADMM_STEP_SIZE * (x_totals - z_vars)

            # 4. Environment Step
            next_obs, rewards, terminated, truncated, _ = env.step(actions, z_vars, y_vars)
            done = terminated or truncated

            # 5. Store joint transition in the multi-agent replay buffer
            replay_buffer.store(obs, actions, rewards, next_obs, done)

            obs = next_obs

            # 6. Perform learning
            if total_timesteps >= START_TIMESTEPS:
                if replay_buffer.size > BATCH_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    maddpg_coordinator.update(batch)

            episode_reward += sum(rewards.values())
            total_timesteps += 1

        # Log and print episode results
        episode_rewards_log.append(episode_reward)
        avg_reward = np.mean(episode_rewards_log[-10:])
        print(f"Episode: {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")

    print("\n--- Training Complete ---")

    # --- Plotting Results ---
    print("\nPlotting training rewards...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(NUM_EPISODES), episode_rewards_log)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode during Training")
    plt.grid(True)
    plt.savefig("training_rewards.png")
    print("Plot saved to training_rewards.png")

    # --- Evaluation Phase ---
    print("\n--- Starting Evaluation Phase ---")
    from baselines import run_static_allocation, run_proportional_fair

    # 1. Evaluate the trained MADDPG agent
    print("\n--- Evaluating Trained MADDPG Agent ---")
    eval_env = NetworkSlicingEnv(rho=ADMM_STEP_SIZE)
    obs, _ = eval_env.reset(seed=42) # Use a fixed seed for fair comparison
    maddpg_reward = 0
    z_vars = np.zeros(num_slices)
    y_vars = np.zeros(num_slices)
    for _ in range(EPISODE_LENGTH):
        actions = maddpg_coordinator.get_actions(obs, noise_std=0) # No exploration noise
        x_totals = np.array([np.sum(actions[f"slice_{i}"]) for i in range(num_slices)])
        z_vars = master_optimizer.solve(x_totals, y_vars)
        if z_vars is None: z_vars = np.zeros(num_slices)
        y_vars += ADMM_STEP_SIZE * (x_totals - z_vars)
        obs, rewards, _, _, _ = eval_env.step(actions, z_vars, y_vars)
        maddpg_reward += sum(rewards.values())
    print(f"Trained MADDPG Agent Total Reward: {maddpg_reward:.2f}")

    # 2. Evaluate baselines
    baseline_env = NetworkSlicingEnv(rho=ADMM_STEP_SIZE)
    baseline_env.reset(seed=42) # Use the same seed
    static_reward = run_static_allocation(baseline_env, EPISODE_LENGTH)

    baseline_env.reset(seed=42) # Use the same seed
    pf_reward = run_proportional_fair(baseline_env, EPISODE_LENGTH)

    # 3. Final comparison
    print("\n--- Evaluation Summary ---")
    print(f"Trained MADDPG Agent: {maddpg_reward:.2f}")
    print(f"Static Allocation:    {static_reward:.2f}")
    print(f"Proportional Fair:    {pf_reward:.2f}")


if __name__ == '__main__':
    main()
