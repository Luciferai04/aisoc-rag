import numpy as np

class TrafficGenerator:
    """
    Generates time-varying traffic demands for different network slices.
    """
    def __init__(self, num_slices, num_users_per_slice):
        """
        Initializes the traffic generator.

        Args:
            num_slices (int): The number of network slices.
            num_users_per_slice (list[int]): A list containing the number of users for each slice.
        """
        self.num_slices = num_slices
        self.num_users_per_slice = num_users_per_slice
        self.time_step = 0

        # Define traffic profiles for different slice types.
        # These profiles determine the characteristics of the traffic generated.
        # (e.g., base demand, fluctuation amplitude, period, and noise level)
        self.slice_profiles = {
            # eMBB: High throughput, significant daily variation
            0: {'name': 'eMBB', 'base': 50, 'amplitude': 40, 'period': 240, 'noise_std': 5},
            # URLLC: Low but critical traffic, less predictable
            1: {'name': 'URLLC', 'base': 5, 'amplitude': 4, 'period': 120, 'noise_std': 2},
            # mMTC: Low-rate, massive number of devices, sporadic
            2: {'name': 'mMTC', 'base': 1, 'amplitude': 0.5, 'period': 300, 'noise_std': 0.5},
        }

        if num_slices > len(self.slice_profiles):
            raise ValueError(f"This traffic generator is configured for {len(self.slice_profiles)} slices, "
                             f"but {num_slices} were requested.")

    def step(self):
        """
        Generates the traffic demand for the current time step for all users in all slices.

        Returns:
            list[np.array]: A list of numpy arrays, where each array contains the
                            traffic demand for each user in a slice.
        """
        all_demands = []
        for i in range(self.num_slices):
            profile = self.slice_profiles[i]
            num_users = self.num_users_per_slice[i]

            # Calculate the sinusoidal part of the traffic
            cyclical_part = profile['amplitude'] * np.sin(2 * np.pi * self.time_step / profile['period'])

            # Generate random noise for each user
            noise = np.random.normal(0, profile['noise_std'], num_users)

            # Calculate the demand for each user in the slice
            demand = profile['base'] + cyclical_part + noise

            # Ensure demand is non-negative
            user_demands = np.maximum(0, demand)
            all_demands.append(user_demands)

        self.time_step += 1
        return all_demands

if __name__ == '__main__':
    # Example usage:
    num_slices = 3
    users_per_slice = [10, 5, 50]  # eMBB, URLLC, mMTC
    traffic_gen = TrafficGenerator(num_slices, users_per_slice)

    # Print demands for a few time steps
    for t in range(5):
        demands = traffic_gen.step()
        print(f"--- Time Step {t} ---")
        for i, user_demands in enumerate(demands):
            profile_name = traffic_gen.slice_profiles[i]['name']
            print(f"  Slice {i} ({profile_name}):")
            print(f"    Avg Demand: {np.mean(user_demands):.2f}")
            print(f"    Total Demand: {np.sum(user_demands):.2f}")
            # print(f"    Sample user demands: {user_demands[:3]}")
        print("\\n")
