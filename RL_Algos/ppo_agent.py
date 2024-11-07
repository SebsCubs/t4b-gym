import torch
from RL_Algos.networks import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bound, lr=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.policy = PolicyNetwork(state_dim, action_dim, action_bound)
        self.policy_old = PolicyNetwork(state_dim, action_dim, action_bound)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = self.policy_old(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.policy.action_bound['low'], self.policy.action_bound['high'])
        action_logprob = dist.log_prob(action).sum()
        return action.numpy(), action_logprob.numpy()

    def update(self, memory_dict):
        # Convert lists to tensors
        states = torch.FloatTensor(memory_dict['states'])
        actions = torch.FloatTensor(memory_dict['actions'])
        rewards = torch.FloatTensor(memory_dict['rewards'])
        old_logprobs = torch.FloatTensor(memory_dict['logprobs'])

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # TODO: Check how to calculate advantages and what the value net should output
        advantages = returns - self.value_net(states).detach().squeeze()

        # PPO policy update
        for _ in range(self.K_epochs):
            mean, std = self.policy(states)
            dist = torch.distributions.Normal(mean, std)
            logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Loss function
            loss = -torch.min(surr1, surr2) + 0.5 * (returns - self.value_net(states).squeeze()) ** 2 - 0.01 * entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Return dictionary of loss statistics
        return {
            'policy_loss': loss.item(),
            'value_loss': (returns - self.value_net(states).squeeze()).pow(2).mean().item(),
            'total_loss': loss.item(),
            'entropy': entropy.item()
        }

# Memory class to store experiences
class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []

    def add(self, state, action, reward, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
