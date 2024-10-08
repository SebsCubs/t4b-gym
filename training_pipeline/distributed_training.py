#TODO: Not finished

import ray
from RL_Algos.ppo_agent import PPOAgent

ray.init()

# Modify BoptestEnv to accept a port number
class BoptestEnv:
    def __init__(self, port=5000):
        self.base_url = f'http://localhost:{port}'
        # Rest of the initialization

# In the distributed rollout function
@ray.remote
def rollout(agent_state_dict, value_state_dict, port):
    env = BoptestEnv(port=port)
    # Load agent and value network state dicts
    agent = PPOAgent(...)
    agent.policy.load_state_dict(agent_state_dict)
    agent.value_net.load_state_dict(value_state_dict)
    # Perform rollout
    # Return experiences

