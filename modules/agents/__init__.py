REGISTRY = {}
from .DQN_agent import DQN_Agent
from .PPO_agent import PPO_Agent
REGISTRY['DQN'] = DQN_Agent
REGISTRY['PPO'] = PPO_Agent

