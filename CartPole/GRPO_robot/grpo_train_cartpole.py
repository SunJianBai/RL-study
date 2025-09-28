# 常用库
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
# PyTorch相关
import torch
from torch.nn import functional as F
import gym
from torch.distributions import Categorical
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)  

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits,dim=1)
    

def collect_trajectory_vectorized(envs, policy_net, trajectory_max_steps=500, device="cpu"):
    """
    从 vectorized 环境并行采样多个轨迹，计算归一化奖励
    envs: 并行环境 (vectorized environment)
    policy_net: 策略网络 (PolicyNet)
    trajectory_max_steps: 每条轨迹最大步数

    返回：
        states, log_probs, actions, normalized_rewards
    """
    group_size = envs.num_envs  # 获取并行环境数量
    seed_num = np.random.randint(0, 1000)
    states = envs.reset(seed = [seed_num] * group_size)  # shape: [group_size, state_dim]

    all_states = []
    all_actions = []
    all_log_probs = []
    all_rewards = torch.zeros(group_size)  # shape: [group_size]
    all_dones = torch.tensor([False] * group_size)  # shape: [group_size]
    for t in range(trajectory_max_steps):
        states_tensor = torch.tensor(states, dtype=torch.float32,device=device)  # shape: [group_size, state_dim]
        probs = policy_net(states_tensor)  # shape: [group_size, num_actions]
        dist = Categorical(probs)
        actions = dist.sample()  # shape: [group_size]
        log_probs = dist.log_prob(actions).detach()

        # 执行环境步进
        next_states, rewards, dones, infos = envs.step(actions.cpu().numpy())

        all_states.append(states)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_dones[dones] = True # 如果环境结束，则标记为True
        rewards[all_dones] = 0 # 如果环境结束，则奖励为0
        rewards += -abs(next_states[:, 0])   # 添加惩罚项，使小车尽量靠近中心
        all_rewards += rewards  # shape: [group_size]

        states = next_states
        if torch.all(all_dones):  # 如果所有环境都结束，则停止
            break
    normalized_rewards = (all_rewards / trajectory_max_steps).to(device)
    all_states = torch.tensor(all_states).permute(1,0,2).to(device)
    all_log_probs = torch.stack(all_log_probs).permute(1,0).to(device)
    all_actions = torch.stack(all_actions).permute(1,0).to(device)
    trajectories = {"all_states":all_states,"all_log_probs": all_log_probs,
                    "all_actions": all_actions,"normalized_rewards": normalized_rewards}
    episode_rewards = (normalized_rewards * trajectory_max_steps)
    return trajectories,episode_rewards

def calc_advantages_with_grpo(trajectories):
    """从轨迹中提取奖励，并标准化每个episode的奖励"""
    rewards = trajectories["normalized_rewards"]  # 提取最终奖励，
    mean_reward = torch.mean(rewards)     # 计算平均值，
    std_reward = torch.std(rewards)  + 1e-8          # 计算标准差（1e-8是防止0除），
    advantages = (rewards - mean_reward)/std_reward  # 最后标准化每个episode

    return advantages

def grpo_update(trajectories, net, optimizer, n_iterations=20, eps=0.2):

    # [1] 使用GRPO函数计算每个episode的标准化Advantage
    advantages = calc_advantages_with_grpo(trajectories).unsqueeze(-1)
    # 将所有轨迹的数据合并成批处理
    all_states = trajectories["all_states"]
    all_log_probs = trajectories["all_log_probs"]
    all_chosen_actions = trajectories["all_actions"]
    batch_size = len(all_states)
    # [2] 更新Policy NN。进行n_iterations次更新
    for i_iter in range(n_iterations):
        loss = 0
        # # Method 1,没有收敛,原因不明
        # # 计算新的对数概率
        # new_log_probs = torch.log(net(all_states).gather(-1, all_chosen_actions.unsqueeze(-1))).squeeze(-1)

        # # 计算概率比
        # ratio = torch.exp(new_log_probs - all_log_probs)

        # # 计算两个 surrogate 项
        # surr1 = ratio * advantages
        # surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

        # # 计算每个步骤的损失
        # trajectory_loss = torch.mean(-torch.min(surr1, surr2), dim=-1)

        # # 计算总损失
        # loss = torch.mean(trajectory_loss)
        # Method 2
        for i in range(batch_size):
            states = all_states[i]
            log_probs = all_log_probs[i]
            chosen_actions = all_chosen_actions[i]
            advantage = advantages[i]
            trajectory_loss = 0                         
            new_log_probs = torch.log(net(states).gather(1,chosen_actions.unsqueeze(1)))
            ratio = torch.exp(new_log_probs - log_probs.unsqueeze(1))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps,
                                1 + eps) * advantage  # 截断
            trajectory_loss = torch.mean(-torch.min(surr1, surr2))       
            loss += trajectory_loss

        # [3] 用episode数进行归一化
        loss /= batch_size

        # [4] 更新Policy NN的权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item() 

# [1] 初始化和初始设置
group_size = 10
env_name = 'CartPole-v1'
envs = gym.vector.make(env_name,num_envs=group_size)
state_dim = envs.single_observation_space.shape[0]  # 4
n_actions = envs.single_action_space.n  # 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
policy = PolicyNet(state_dim, n_actions).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
episode_num = 50  # 尝试次数
trajectory_max_steps = 500 #每条轨迹最长的探索步数
return_list = []

start = time.time()
# [2] 开始50次episode循环
for i_episode in tqdm(range(episode_num)):  
    # [3] 使用GRPO积累轨迹（10次episode）
    trajectories,episode_rewards = collect_trajectory_vectorized(envs,policy,trajectory_max_steps,device=device)

    # [4] 使用GRPO更新PolicyNet的权重
    loss = grpo_update(trajectories, policy, optimizer)
    
    # [5] 计算平均奖励
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    return_list.append(avg_reward.cpu().numpy())
    # if i_episode !=0 and i_episode % 20 == 0:
    #     save_path = f"grpo_cartpole_policy_update_{i_episode}.pth"
    #     torch.save(policy.state_dict(), save_path)
    #     print(f"Model saved to {save_path}")
    print(f'第 {i_episode} 次试验, avg reward: {avg_reward:.2f}')    
    # # [7] 提前结束判定
    # if avg_reward > trajectory_max_steps-5:
    #     print('训练完成。试验次数: ', i_episode)
    #     break
print("used_time(s): ", time.time() - start)

save_path = f"./weights/grpo_cartpole_policy_update_final.pth"
torch.save(policy.state_dict(), save_path)
print(f"Model saved to {save_path}")
#绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GRPO on {}'.format(env_name))
plt.grid(True)
plt.show()
envs.close()
