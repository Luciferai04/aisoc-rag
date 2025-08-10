import numpy as np

def run_static_allocation(env, episode_length=100):
    """
    Runs a baseline with a fixed, static resource allocation.

    Args:
        env (gym.Env): The network slicing environment.
        episode_length (int): The number of timesteps to run the evaluation for.

    Returns:
        float: The total reward achieved over the episode.
    """
    print("\n--- Running Static Allocation Baseline ---")
    obs, _ = env.reset()
    total_reward = 0

    # Define a fixed allocation (example: 40% eMBB, 20% URLLC, 40% mMTC)
    slice_allocations = np.array([0.4, 0.2, 0.4]) * env.total_resources

    actions = {}
    for i in range(env.num_slices):
        num_users = env.num_users_per_slice[i]
        # Distribute the slice's total allocation equally among its users
        actions[f'slice_{i}'] = np.full(num_users, slice_allocations[i] / num_users)

    # Dummy ADMM variables, as this baseline does not use them
    z_vars = slice_allocations
    y_vars = np.zeros(env.num_slices)

    for t in range(episode_length):
        # The action is static, so it does not change
        next_obs, rewards, terminated, truncated, _ = env.step(actions, z_vars, y_vars)
        total_reward += sum(rewards.values())
        if terminated or truncated:
            break

    print(f"Static Allocation Total Reward: {total_reward:.2f}")
    return total_reward


def run_proportional_fair(env, episode_length=100):
    """
    Runs a baseline that allocates resources proportionally to user demand.

    Args:
        env (gym.Env): The network slicing environment.
        episode_length (int): The number of timesteps to run the evaluation for.

    Returns:
        float: The total reward achieved over the episode.
    """
    print("\n--- Running Proportional Fair Baseline ---")
    obs, _ = env.reset()
    total_reward = 0

    # Dummy ADMM variables
    z_vars = np.zeros(env.num_slices)
    y_vars = np.zeros(env.num_slices)

    for t in range(episode_length):
        # Get current demands from the environment state
        current_demands_flat = np.concatenate([obs[f'slice_{i}'][:env.num_users_per_slice[i]] for i in range(env.num_slices)])

        total_demand = np.sum(current_demands_flat)
        if total_demand == 0:
            # Avoid division by zero if there's no demand
            proportions = np.ones_like(current_demands_flat) / len(current_demands_flat)
        else:
            proportions = current_demands_flat / total_demand

        allocations_flat = proportions * env.total_resources

        # Reconstruct the actions dictionary
        actions = {}
        user_idx = 0
        for i in range(env.num_slices):
            num_users = env.num_users_per_slice[i]
            actions[f'slice_{i}'] = allocations_flat[user_idx : user_idx + num_users]
            user_idx += num_users

        next_obs, rewards, terminated, truncated, _ = env.step(actions, z_vars, y_vars)
        obs = next_obs
        total_reward += sum(rewards.values())
        if terminated or truncated:
            break

    print(f"Proportional Fair Total Reward: {total_reward:.2f}")
    return total_reward
