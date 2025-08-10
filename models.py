import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor Network for the DDPG Agent.
    Maps states to actions.
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): The maximum value an action can take. Used to scale the output.
        """
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The output action, scaled to the environment's action range.
        """
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # Use tanh to bound the action output to [-1, 1]
        x = torch.tanh(self.layer_3(x))
        # Scale the action to [0, max_action]. Assumes actions are non-negative.
        return (x + 1) / 2 * self.max_action


class Critic(nn.Module):
    """
    Critic Network for the DDPG Agent (for independent learners).
    Estimates the Q-value for a given state-action pair for a single agent.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the Critic network.

        Args:
            state_dim (int): Dimension of the state space for a single agent.
            action_dim (int): Dimension of the action space for a single agent.
        """
        super(Critic, self).__init__()

        # Layer 1 processes the state
        self.layer_1 = nn.Linear(state_dim, 256)
        # Layer 2 combines the processed state and the action
        self.layer_2 = nn.Linear(256 + action_dim, 256)
        # Layer 3 outputs the Q-value
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state for a single agent.
            action (torch.Tensor): The input action for a single agent.

        Returns:
            torch.Tensor: The estimated Q-value.
        """
        # Process state first
        state_out = F.relu(self.layer_1(state))
        # Concatenate processed state with the action
        combined = torch.cat([state_out, action], 1)
        # Further processing
        x = F.relu(self.layer_2(combined))
        q_value = self.layer_3(x)
        return q_value


class CentralizedCritic(nn.Module):
    """
    Centralized Critic Network for MADDPG.
    Estimates the Q-value based on the joint state and actions of all agents.
    """
    def __init__(self, total_state_dim, total_action_dim):
        """
        Initializes the Centralized Critic network.

        Args:
            total_state_dim (int): Sum of state dimensions of all agents.
            total_action_dim (int): Sum of action dimensions of all agents.
        """
        super(CentralizedCritic, self).__init__()

        self.layer_1 = nn.Linear(total_state_dim + total_action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, all_states, all_actions):
        """
        Performs the forward pass.

        Args:
            all_states (torch.Tensor): Concatenated states of all agents.
            all_actions (torch.Tensor): Concatenated actions of all agents.

        Returns:
            torch.Tensor: The estimated joint Q-value.
        """
        combined_input = torch.cat([all_states, all_actions], 1)
        x = F.relu(self.layer_1(combined_input))
        x = F.relu(self.layer_2(x))
        q_value = self.layer_3(x)
        return q_value

if __name__ == '__main__':
    # Example usage and sanity check
    state_dim = 21 # Example from environment.py: 10 users * 2 (demand + qos) + 1 (admm)
    action_dim = 10
    max_action = 1000

    # Create actor and critic
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)

    print("--- Actor Network ---")
    print(actor)

    print("\n--- Critic Network ---")
    print(critic)

    # Create a dummy state and action
    dummy_state = torch.randn(1, state_dim)
    dummy_action_from_actor = actor(dummy_state)

    print(f"\nDummy state shape: {dummy_state.shape}")
    print(f"Dummy action shape from actor: {dummy_action_from_actor.shape}")

    # Get a Q-value from the critic
    q_value = critic(dummy_state, dummy_action_from_actor)
    print(f"Dummy Q-value from critic: {q_value.item():.2f}")

    assert dummy_action_from_actor.shape == (1, action_dim)
    assert q_value.shape == (1, 1)
    print("\nModel dimensions are correct.")
