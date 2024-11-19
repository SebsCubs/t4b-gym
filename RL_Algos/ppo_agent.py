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

        # Normalize advantages
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Safer normalization with clipping
        returns = torch.clamp(returns, -10.0, 10.0)  # Clip extreme returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        values = self.value_net(states).detach().squeeze()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        # PPO policy update
        for _ in range(self.K_epochs):
            # Get current policy distribution
            mean, std = self.policy(states)
            
            # Clamp mean and std to prevent extreme values
            mean = torch.clamp(mean, -2.0, 2.0)
            std = torch.clamp(std, 0.1, 0.5)
            
            try:
                dist = torch.distributions.Normal(mean, std)
                logprobs = dist.log_prob(actions)
                logprobs = torch.clamp(logprobs, -20.0, 2.0)  # Prevent extreme log probs
                logprobs = logprobs.sum(dim=1)
                
                entropy = dist.entropy().sum(dim=1)
                #entropy = torch.clamp(entropy, -1.0, 1.0)  # Clip entropy
                
                ratios = torch.exp(torch.clamp(logprobs - old_logprobs, -20.0, 2.0))
                ratios = torch.clamp(ratios, 0.0, 4.0)  # Prevent extreme ratios

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Value loss with clipping
                value_pred = self.value_net(states).squeeze()
                value_loss = 0.5 * torch.clamp((returns - value_pred) ** 2, 0.0, 10.0)

                # Combined loss with careful scaling
                policy_loss = -torch.min(surr1, surr2)
                loss = (policy_loss + 0.5 * value_loss - 0.01 * entropy).mean()

                # Check for invalid values before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Invalid loss detected, skipping update")
                    continue

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.optimizer.step()

            except Exception as e:
                print(f"Error in policy update: {str(e)}")
                continue

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Return dictionary of loss statistics
        return {
            'policy_loss': policy_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'total_loss': loss.item(),
            'entropy': entropy.mean().item(),
            'training_step': self.K_epochs
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
