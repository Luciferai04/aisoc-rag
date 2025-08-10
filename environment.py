import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import TrafficGenerator

class NetworkSlicingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for 5G Network Slicing.

    This environment simulates multiple network slices (e.g., eMBB, URLLC, mMTC)
    with time-varying traffic, and asks RL agents to dynamically allocate
    resources to users within each slice.

    The environment is designed to work with the DeepSlicing framework, which
    combines DRL with ADMM for coordination.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_slices=3, num_users_per_slice=[10, 5, 50], total_resources=1000, rho=0.5, beta=10.0):
        super(NetworkSlicingEnv, self).__init__()

        self.num_slices = num_slices
        self.num_users_per_slice = num_users_per_slice
        self.total_users = sum(num_users_per_slice)
        self.total_resources = total_resources

        # ADMM and Reward hyperparameters
        self.rho = rho  # Penalty parameter for ADMM
        self.beta = beta # Penalty for QoS violation

        # Initialize the traffic generator
        self.traffic_gen = TrafficGenerator(self.num_slices, self.num_users_per_slice)

        # Define QoS requirements for each slice type (Utility minimums)
        self.slice_qos_reqs = {
            0: {'name': 'eMBB', 'min_utility': 3.5}, # Corresponds to ~30 Mbps if U=log(1+thr)
            1: {'name': 'URLLC', 'min_utility': 2.0}, # Corresponds to ~6 Mbps
            2: {'name': 'mMTC', 'min_utility': 0.6}, # Corresponds to ~0.8 Mbps
        }
        self.min_utilities = [np.full(num_users, self.slice_qos_reqs[i]['min_utility'])
                              for i, num_users in enumerate(self.num_users_per_slice)]

        # Define action space: one continuous action per user (resource allocation)
        # We define it as a dictionary for multi-agent compatibility
        self.action_space = spaces.Dict({
            f"slice_{i}": spaces.Box(low=0, high=self.total_resources, shape=(num_users,), dtype=np.float32)
            for i, num_users in enumerate(self.num_users_per_slice)
        })

        # Define observation space (state):
        # For each slice: [current_traffic_demands, prev_qos_satisfaction, admm_variable]
        self.observation_space = spaces.Dict({
            f"slice_{i}": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_users * 2 + 1,), # demands + qos_satisfactions + (z_i - y_i)
                dtype=np.float32
            )
            for i, num_users in enumerate(self.num_users_per_slice)
        })

    def _get_obs(self):
        """Constructs the observation for each agent."""
        observations = {}
        for i in range(self.num_slices):
            # Ensure all components have consistent dtypes and are 1D arrays
            demands = self.current_demands[i].flatten().astype(np.float32)
            qos_satisfaction = self.prev_qos_satisfaction[i].flatten().astype(np.float32)
            admm_var = np.array([self.z_vars[i] - self.y_vars[i]], dtype=np.float32)

            observations[f"slice_{i}"] = np.concatenate([demands, qos_satisfaction, admm_var])
        return observations

    def _calculate_utility(self, demands, allocations):
        """Calculates utility based on demand and allocation.
        Utility is based on the throughput achieved.
        """
        throughput = np.minimum(demands, allocations)
        # Use log utility to model diminishing returns
        return np.log1p(throughput)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset traffic generator and get initial demands
        self.traffic_gen = TrafficGenerator(self.num_slices, self.num_users_per_slice)
        self.current_demands = self.traffic_gen.step()

        # Reset internal state
        self.prev_qos_satisfaction = [np.zeros(n_users) for n_users in self.num_users_per_slice]
        self.z_vars = np.zeros(self.num_slices)
        self.y_vars = np.zeros(self.num_slices)

        # Initial observation
        observation = self._get_obs()
        info = {} # No extra info on reset

        return observation, info

    def step(self, actions, z_vars, y_vars):
        """
        Executes one time step in the environment.

        Args:
            actions (dict): A dictionary of actions from each agent.
            z_vars (np.array): ADMM global variables.
            y_vars (np.array): ADMM dual variables.
        """
        self.z_vars = z_vars
        self.y_vars = y_vars

        rewards = {}

        for i in range(self.num_slices):
            slice_actions = actions[f"slice_{i}"]
            slice_demands = self.current_demands[i]
            slice_min_utils = self.min_utilities[i]

            # 1. Calculate utility for each user in the slice
            utilities = self._calculate_utility(slice_demands, slice_actions)

            # 2. Calculate the reward for the slice
            # Part 1: Sum of utilities + penalty for QoS violation
            qos_satisfaction = utilities / slice_min_utils
            qos_violation = np.maximum(0, 1 - qos_satisfaction)
            reward_part1 = np.sum(utilities - self.beta * qos_violation)

            # Part 2: ADMM penalty for deviation from global consensus
            total_slice_allocation = np.sum(slice_actions)
            admm_penalty = (self.rho / 2) * np.square(total_slice_allocation - self.z_vars[i] + self.y_vars[i])

            rewards[f"slice_{i}"] = reward_part1 - admm_penalty

            # Update state for next observation
            self.prev_qos_satisfaction[i] = qos_satisfaction

        # Update to next time step's demands
        self.current_demands = self.traffic_gen.step()

        # Get next observation
        observation = self._get_obs()

        # For simplicity, we run for a fixed number of steps, so terminated/truncated are always False
        terminated = False
        truncated = False
        info = {} # Can be used to return debug info

        return observation, rewards, terminated, truncated, info

if __name__ == '__main__':
    # Example of how to use the environment
    env = NetworkSlicingEnv()
    obs, info = env.reset()

    print("--- Initial Observation ---")
    for slice_name, slice_obs in obs.items():
        print(f"  {slice_name}: shape={slice_obs.shape}, dtype={slice_obs.dtype}")

    # Example step with random actions and dummy ADMM variables
    actions = env.action_space.sample()
    z_vars = np.random.rand(env.num_slices) * 300
    y_vars = np.random.rand(env.num_slices) * 10

    print("\n--- Taking a Step ---")
    next_obs, rewards, _, _, _ = env.step(actions, z_vars, y_vars)

    print("\n--- Step Result ---")
    print("Next Observation:")
    for slice_name, slice_obs in next_obs.items():
        print(f"  {slice_name}: shape={slice_obs.shape}")

    print("\nRewards:")
    for slice_name, reward in rewards.items():
        print(f"  {slice_name}: {reward:.2f}")
