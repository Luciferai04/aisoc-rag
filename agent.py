import torch
import torch.optim as optim
import numpy as np
from models import Actor, Critic, CentralizedCritic

# --- INDEPENDENT DDPG AGENT (from previous step) ---

class IDDPGAgent:
    """The Independent Deep Deterministic Policy Gradient Agent."""
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def get_action(self, state, noise_std=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * noise_std, size=action.shape)
        return (action + noise).clip(0, self.max_action)

    def update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# --- MADDPG IMPLEMENTATION ---

class MAReplayBuffer:
    """A replay buffer for multiple agents."""
    def __init__(self, num_agents, state_dims, action_dims, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states = [np.zeros((max_size, dim)) for dim in state_dims]
        self.actions = [np.zeros((max_size, dim)) for dim in action_dims]
        self.rewards = [np.zeros((max_size, 1)) for _ in range(num_agents)]
        self.next_states = [np.zeros((max_size, dim)) for dim in state_dims]
        self.dones = np.zeros((max_size, 1))

    def store(self, states, actions, rewards, next_states, done):
        for i in range(self.num_agents):
            self.states[i][self.ptr] = states[f'slice_{i}']
            self.actions[i][self.ptr] = actions[f'slice_{i}']
            self.rewards[i][self.ptr] = rewards[f'slice_{i}']
            self.next_states[i][self.ptr] = next_states[f'slice_{i}']
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            [torch.FloatTensor(self.states[i][ind]).to(self.device) for i in range(self.num_agents)],
            [torch.FloatTensor(self.actions[i][ind]).to(self.device) for i in range(self.num_agents)],
            [torch.FloatTensor(self.rewards[i][ind]).to(self.device) for i in range(self.num_agents)],
            [torch.FloatTensor(self.next_states[i][ind]).to(self.device) for i in range(self.num_agents)],
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )

class MADDPG:
    """The Multi-Agent DDPG Coordinator."""
    def __init__(self, agents, total_state_dim, total_action_dim, lr_critic=1e-3):
        self.agents = agents
        self.num_agents = len(agents)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create centralized critic
        self.critic = CentralizedCritic(total_state_dim, total_action_dim).to(self.device)
        self.critic_target = CentralizedCritic(total_state_dim, total_action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update(self, batch, gamma=0.99):
        states, actions, rewards, next_states, dones = batch

        # --- Update Critic ---
        with torch.no_grad():
            next_actions = torch.cat([agent.actor_target(next_states[i]) for i, agent in enumerate(self.agents)], dim=1)
            target_q = self.critic_target(torch.cat(next_states, dim=1), next_actions)
            # Assumes rewards is a list of tensors [agent_0_rewards, agent_1_rewards, ...]
            # We use the sum of rewards for the joint Q value
            total_reward = torch.cat(rewards, dim=1).sum(dim=1, keepdim=True)
            target_q = total_reward + (1 - dones) * gamma * target_q

        current_q = self.critic(torch.cat(states, dim=1), torch.cat(actions, dim=1))

        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actors ---
        for i, agent in enumerate(self.agents):
            # Create a version of all actions where agent i's action is taken from its policy
            current_actions_for_actor_loss = actions.copy()
            current_actions_for_actor_loss[i] = agent.actor(states[i])

            actor_loss = -self.critic(torch.cat(states, dim=1), torch.cat(current_actions_for_actor_loss, dim=1)).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        # --- Update Target Networks ---
        self.update_target_networks()

    def update_target_networks(self):
        # Soft update for the centralized critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.agents[0].tau * param.data + (1.0 - self.agents[0].tau) * target_param.data)
        # Soft update for each agent's actor
        for agent in self.agents:
            agent.update_target_networks() # This will update both actor and its independent critic, which is fine.

    def get_actions(self, states, noise_std=0.1):
        actions = {}
        for i, agent in enumerate(self.agents):
            actions[f'slice_{i}'] = agent.get_action(states[f'slice_{i}'], noise_std)
        return actions
