from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt
plt.style.use('dark_background')
import datetime
from minesweeper_env import *

# DQN Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
NUM_EPISODES = 5000

def save_model_to_file(model, name):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models\\{NUM_EPISODES}-{name}-{timestamp}.pth')

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.steps_done = 0
        self.output_size = output_size


    def select_action(self, state, epsilon, cover_grid):
        valid_actions = [i for i in range(self.output_size) if cover_grid[i // NUM_COLS][i % NUM_COLS] == -3] #-3 means tile is unrevealed. testing issues with ambiguity between cover_grid[r][c] = 0 (unrevealed tile) and grid[r][c] = 0 (revealed tile with 0 bombs nearby) in the merged_grid
        assert len(valid_actions) > 0, "No valid actions available"
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.policy_net(state)
            q_values = q_values.detach().numpy().flatten()
            q_values = [(q_values[i], i) for i in valid_actions]
            q_values.sort(reverse=True)
            action = q_values[0][1]
        print(f'Action: {action}, Q-value: {q_values[0][0]}, Valid actions: {valid_actions}')
        return action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(state_batch).view(BATCH_SIZE, -1)
        action_batch = torch.LongTensor(action_batch).view(-1,1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.cat(next_state_batch).view(BATCH_SIZE, -1)
        done_batch = torch.FloatTensor(done_batch)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        print(q_values[:5])
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + GAMMA * max_next_q_values * (1 - done_batch)

        # print(f"state_batch shape: {state_batch.shape}")
        # print(f"action_batch shape: {action_batch.shape}")
        # print(f"q_values shape: {q_values.shape}")
        # print(f"target_q_values shape: {target_q_values.shape}")
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        print(f'Loss: {loss.item()}')

def train_dqn(env, agent, num_episodes=NUM_EPISODES):
    epsilon = EPS_START
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_w and (pygame.key.get_mods() & pygame.KMOD_META)):
                    pygame.quit()
                    return
                
            action = agent.select_action(state, epsilon, env.merged_grid)
            next_state, reward, done, won = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.optimize_model()
            # env.render()

            pygame.time.Clock().tick(30)

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards_per_episode.append((total_reward, won))


    if episode == num_episodes - 1:
        save_model_to_file(agent.policy_net, 'policy_net')
        save_model_to_file(agent.target_net, 'target_net')
        print('Saved to file')

    print('Ilosc zwyciestw:', sum([x[1] for x in rewards_per_episode]))
    plt.figure(figsize=(12,12))
    plt.scatter([x for x in range(len(rewards_per_episode))], [x[0] for x in rewards_per_episode], c= ['green' if x[1] else 'red' for x in rewards_per_episode ])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Minesweeper')
    plt.grid(False)
    plt.savefig(f'plots\\dqn_minesweeper-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png')
    plt.show()

    moving_avg_rewards = [sum([x[0] for x in rewards_per_episode[max(0, i-50):i]]) / 50 for i in range(len(rewards_per_episode))]
    plt.plot(moving_avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Reward Trend')
    plt.savefig(f'plots\\dqn_minesweeper-moving-average-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png')
    plt.show()


if __name__ == "__main__":
    input_dim = NUM_ROWS * NUM_COLS
    output_dim = NUM_ROWS * NUM_COLS

    env = MinesweeperEnv()
    agent = DQNAgent(input_size=input_dim, output_size=output_dim)
    train_dqn(env, agent)

