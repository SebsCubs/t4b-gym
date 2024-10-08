import time
from boptest_env import BoptestEnv
from RL_Algos.ppo_agent import PPOAgent, Memory
from torch.utils.tensorboard import SummaryWriter
import torch


def train():
    env = BoptestEnv()
    state_dim = len(env.get_observation())
    action_dim = len(env.action_space['low'])
    action_bound = {'low': env.action_space['low'], 'high': env.action_space['high']}

    agent = PPOAgent(state_dim, action_dim, action_bound)
    memory = Memory()

    max_episodes = 1000
    max_timesteps = 200
    log_interval = 10  # Print every n episodes
    save_interval = 30

    writer = SummaryWriter('runs/boptest_experiment')

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            action, action_logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, reward, action_logprob)
            state = next_state
            episode_reward += reward
            if done:
                break
        
     
        # Update the agent
        agent.update(memory)
        memory.clear()
        # Saving checkpoint
        if episode % save_interval == 0:
            torch.save(agent.policy.state_dict(), f'checkpoints/policy_ep{episode}.pth')
            torch.save(agent.value_net.state_dict(), f'checkpoints/value_ep{episode}.pth')

        # Logging
        if episode % log_interval == 0:
            writer.add_scalar('Episode Reward', episode_reward, episode)   


if __name__ == '__main__':
    train()

