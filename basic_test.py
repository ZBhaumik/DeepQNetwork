import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import your ReplayBuffer and DQN classes
from replay_buffer import ReplayBuffer
from dqn import DQN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
TAU = 1e-3
LR = 1e-3
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
UPDATE_EVERY = 4

# Initialize environment
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize replay buffer and DQN agent
replay_buffer = ReplayBuffer(state_size, action_size, BUFFER_SIZE)
agent = DQN(state_size, action_size, GAMMA, TAU, LR)

# Training loop
epsilon = EPSILON_START
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0

    while True:
        # Agent selects action using epsilon-greedy policy
        action = agent.act(state, epsilon)
        
        # Take action and observe next state, reward, and done flag
        next_state, reward, done, _, _ = env.step(action)
        
        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update agent every UPDATE_EVERY steps if buffer is sufficiently filled
        if len(replay_buffer) > BATCH_SIZE and episode % UPDATE_EVERY == 0:
            batch = replay_buffer.sample(BATCH_SIZE)
            loss = agent.update(batch)
        
        # Accumulate total reward
        total_reward += reward
        state = next_state
        
        # Break if episode is done
        if done:
            break
    
    # Decay epsilon after each episode
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # Print episode statistics
    print(f"Episode {episode}/{NUM_EPISODES} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save trained model
torch.save(agent.model.state_dict(), 'dqn_model.pth')

# Testing the trained agent
def test_agent(agent, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)  # No exploration during testing
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
        print(f"Test Episode Reward: {episode_reward}")
    print(f"Average Test Reward: {np.mean(total_rewards)}")

# Load the trained model
test_agent(agent, gym.make(ENV_NAME))