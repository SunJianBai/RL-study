from nbformat import v4 as nbf

# 创建一个notebook
nb = nbf.new_notebook()

# Notebook内容
cells = []

# 标题
cells.append(nbf.new_markdown_cell("# DQN 在 CartPole-v1 环境中的实现与可视化"))

# 导入依赖
cells.append(nbf.new_code_cell("""
import gymnasium as gym
import math, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
"""))

# 定义Q网络
cells.append(nbf.new_code_cell("""
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.fc(x)
"""))

# 经验回放
cells.append(nbf.new_code_cell("""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    def __len__(self):
        return len(self.buffer)
"""))

# 训练超参数和函数
cells.append(nbf.new_code_cell("""
def train_dqn(env, num_episodes=300, batch_size=64, gamma=0.99, lr=1e-3,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)

    steps_done = 0
    rewards = []

    def select_action(state):
        nonlocal steps_done
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).max(1)[1].item()
        else:
            return random.randrange(action_dim)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = buffer.sample(batch_size)
                b_state = torch.tensor(b_state, dtype=torch.float32, device=device)
                b_action = torch.tensor(b_action, dtype=torch.int64, device=device).unsqueeze(1)
                b_reward = torch.tensor(b_reward, dtype=torch.float32, device=device).unsqueeze(1)
                b_next_state = torch.tensor(b_next_state, dtype=torch.float32, device=device)
                b_done = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy_net(b_state).gather(1, b_action)
                next_q_values = target_net(b_next_state).max(1)[0].detach().unsqueeze(1)
                expected_q_values = b_reward + gamma * next_q_values * (1 - b_done)

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        rewards.append(total_reward)
        if (episode+1) % 20 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}")
    return policy_net, rewards
"""))

# 设置设备并运行训练
cells.append(nbf.new_code_cell("""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")
policy_net, rewards = train_dqn(env, num_episodes=200)
env.close()
"""))

# 可视化奖励
cells.append(nbf.new_code_cell("""
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards over Episodes")
plt.show()
"""))

# 渲染测试
cells.append(nbf.new_code_cell("""
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).max(1)[1].item()
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
"""))

# 生成notebook
nb["cells"] = cells

with open("DQN_CartPole.ipynb", "w", encoding="utf-8") as f:
    f.write(nbf.writes(nb))
