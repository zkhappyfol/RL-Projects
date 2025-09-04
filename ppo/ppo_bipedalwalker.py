import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt # 引入绘图库
# vvvvvvvvvvvvvv  在这里添加下面两行 vvvvvvvvvvvvvv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



# --- 1. 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 3e-4         # 演员学习率
LR_CRITIC = 1e-3        # 评论家学习率
GAMMA = 0.99            # 折扣因子
GAE_LAMBDA = 0.95       # GAE参数，用于优势函数计算
EPS_CLIP = 0.2          # PPO裁剪范围
K_EPOCHS = 10           # 每次更新时，对同一批数据学习的次数
UPDATE_TIMESTEP = 4000  # 每收集这么多步数据后，进行一次更新

# --- 2. 演员-评论家网络骨架 (与之前类似) ---
class ActorCriticShared(nn.Module):
    def __init__(self, state_dim):
        super(ActorCriticShared, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        return torch.relu(self.layer2(x))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.shared = ActorCriticShared(state_dim)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, state):
        features = self.shared(state)
        action_mean = torch.tanh(self.mean_head(features))
        return action_mean

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.shared = ActorCriticShared(state_dim)
        self.value_head = nn.Linear(256, 1)
    def forward(self, state):
        features = self.shared(state)
        return self.value_head(features)

# --- 3. PPO 智能体 ---
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        # 临时存储一个回合(或多个回合)的数据
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

    def select_action(self, state):
        """根据当前策略选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            action_mean = self.actor(state_tensor)
            action_std = torch.exp(self.actor.log_std).to(DEVICE)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob.cpu().item()

    def update(self):
        """PPO核心更新逻辑"""
        # 1. 事后复盘：计算优势函数 (Advantage)
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # 标准化

        old_states = torch.FloatTensor(np.array(self.memory['states'])).to(DEVICE)
        old_actions = torch.FloatTensor(np.array(self.memory['actions'])).to(DEVICE)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32).to(DEVICE)

        # 2. 温故而知新：对同一批数据进行 K_EPOCHS 次学习
        for _ in range(K_EPOCHS):
            # 评估旧动作在新策略下的情况
            action_mean = self.actor(old_states)
            action_std = torch.exp(self.actor.log_std).to(DEVICE)
            dist = Normal(action_mean, action_std)
            
            new_log_probs = dist.log_prob(old_actions).sum(1)
            entropy = dist.entropy().sum(1)
            state_values = self.critic(old_states)

            # 3. 戴上“安全带”：计算PPO裁剪损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            advantages = rewards - state_values.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            # 最终损失 = 裁剪后的演员损失 + 评论家损失 + 熵奖励
            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * entropy.mean()
            
            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 清空“今日日记”
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

# --- 4. 主训练循环 ---
if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    timestep = 0

    # <--- 新增: 创建一个空列表来存储每个回合的奖励 ---
    all_episode_rewards = []

    for i_episode in range(1, 1001):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            timestep += 1
            
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录“日记”
            agent.memory['states'].append(state)
            agent.memory['actions'].append(action)
            agent.memory['log_probs'].append(log_prob)
            agent.memory['rewards'].append(reward)
            agent.memory['dones'].append(done)
            
            state = next_state
            episode_reward += reward
            
            # 如果收集到足够的数据，就进行一次集中的学习
            if timestep % UPDATE_TIMESTEP == 0:
                agent.update()
        
        # <--- 新增: 将当前回合的奖励存入列表 ---
        all_episode_rewards.append(episode_reward)
        
        print(f'Episode: {i_episode}, Reward: {episode_reward:.2f}')
        
    # <--- 新增: 训练结束后，收尾并绘图 ---
    print("\n训练完成!")
    env.close()

    # 绘制学习曲线
    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_rewards, label='每回合奖励')
    plt.title('PPO on BipedalWalker-v3 训练过程')
    plt.xlabel('回合 (Episode)')
    plt.ylabel('总奖励 (Total Reward)')

    # 绘制最近100个回合的平均奖励曲线，以更好地观察趋势
    if len(all_episode_rewards) >= 100:
        avg_rewards = [np.mean(all_episode_rewards[i:i+100]) for i in range(len(all_episode_rewards) - 100)]
        plt.plot(np.arange(100, len(all_episode_rewards)), avg_rewards, label='最近100回合平均奖励', color='orange', linewidth=3)
    
    plt.legend()
    plt.grid(True)
    plt.show()