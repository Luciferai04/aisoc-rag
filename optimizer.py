import numpy as np
import cvxpy as cp

class MasterOptimizer:
    """
    Solves the ADMM Master Problem using convex optimization (CVXPY).

    The master problem determines the globally consistent resource allocation 'z'
    for each slice, given the proposals 'x' from the individual DRL agents.
    """
    def __init__(self, num_slices, total_resources):
        """
        Initializes the Master Optimizer.

        Args:
            num_slices (int): The number of network slices.
            total_resources (float): The total available network resources.
        """
        self.num_slices = num_slices
        self.total_resources = total_resources

        # Define the optimization variables for CVXPY
        self._z = cp.Variable(num_slices)

        # Define the parameters that will be passed in at solve time
        self._x_totals = cp.Parameter(num_slices)
        self._y_vars = cp.Parameter(num_slices)

        # Define the objective function
        # Objective is to minimize the augmented Lagrangian w.r.t. z.
        # This simplifies to minimizing the quadratic penalty term.
        objective = cp.Minimize(cp.sum_squares(self._x_totals - self._z + self._y_vars))

        # Define the constraints
        constraints = [
            cp.sum(self._z) <= self.total_resources,
            self._z >= 0
        ]

        # Create the CVXPY problem
        self._problem = cp.Problem(objective, constraints)

    def solve(self, x_totals, y_vars):
        """
        Solves the master optimization problem.

        Args:
            x_totals (np.array): A numpy array where x_totals[i] is the sum of
                                 resources proposed by agent i.
            y_vars (np.array): The current dual variables from the ADMM updates.

        Returns:
            np.array: The optimal 'z' values for each slice. Returns None if the
                      problem cannot be solved.
        """
        # Set the parameter values
        self._x_totals.value = x_totals
        self._y_vars.value = y_vars

        # Solve the problem
        # Use a robust solver like SCS, which is good for QPs.
        self._problem.solve(solver=cp.SCS)

        if self._problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return self._z.value
        else:
            print(f"Warning: Master problem could not be solved. Status: {self._problem.status}")
            # As a fallback, return a simple proportional allocation
            return (x_totals / np.sum(x_totals)) * self.total_resources if np.sum(x_totals) > 0 else x_totals


if __name__ == '__main__':
    # Example usage and sanity check
    num_slices = 3
    total_resources = 1000

    optimizer = MasterOptimizer(num_slices, total_resources)

    # Dummy inputs from agents
    # Agents are requesting more than is available
    x_totals_from_agents = np.array([500, 300, 400]) # Total request = 1200
    # Initial dual variables are often zero
    y_vars_initial = np.zeros(num_slices)

    print("--- Solving Master Problem ---")
    print(f"Agent proposals (x_totals): {x_totals_from_agents}")
    print(f"Initial dual variables (y_vars): {y_vars_initial}")

    z_solution = optimizer.solve(x_totals_from_agents, y_vars_initial)

    print(f"\nOptimizer solution (z_vars): {np.round(z_solution, 2)}")
    print(f"Sum of z_vars: {np.sum(z_solution):.2f} (should be <= {total_resources})")

    assert z_solution is not None
    assert np.all(z_solution >= 0)
    assert np.sum(z_solution) <= total_resources + 1e-6 # Add tolerance for float precision

    print("\nMasterOptimizer works as expected.")
