import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


# --- Part 1: 超参数设定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 优先使用GPU
MEMORY_CAPACITY = 10000     # 记忆宫殿的容量
BATCH_SIZE = 128            # 每次学习的批量大小
GAMMA = 0.99                # 折扣因子
TAU = 0.005                 # 目标网络软更新系数
ACTOR_LR = 1e-4             # 演员网络的学习率
CRITIC_LR = 1e-3            # 评论家网络的学习率
NOISE_STDDEV = 0.1          # 动作噪声的标准差

# --- Part 2: 经验回放池 (老朋友) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)

# --- Part 3: 演员网络 (驾驶员) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # 使用 tanh 将输出缩放到 [-1, 1] 之间, 再乘以动作的最大值
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

# --- Part 4: 评论家网络 (导航员) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 接收 state 和 action 作为输入
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1) # 输出一个Q值

    def forward(self, state, action):
        # 将 state 和 action 在维度1上拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Part 5: DDPG 总指挥 ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        
        # 创建四个网络
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict()) # 复制权重
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict()) # 复制权重
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.memory = ReplayBuffer(MEMORY_CAPACITY)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        # 添加噪声以进行探索
        noise = np.random.normal(0, self.max_action * NOISE_STDDEV, size=action.shape)
        return (action + noise).clip(-self.max_action, self.max_action)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # 转换成Tensor
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # --- 更新评论家网络 ---
        # 1. 计算目标Q值
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * GAMMA * target_q
        
        # 2. 计算当前Q值
        current_q = self.critic(states, actions)
        
        # 3. 计算评论家损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 4. 优化评论家
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新演员网络 ---
        # 演员的目标是最大化评论家给出的Q值
        # 所以损失是Q值的负数
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# --- Part 6: 主训练循环 (优化版) ---
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action)

num_episodes = 200

print("="*30)
print(f"开始在 {DEVICE} 上训练...")
print("请耐心等待，第一个回合的报告可能需要一些时间...")
print("="*30)

for i_episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    
    # Pendulum-v1 环境默认在200步后结束
    for t in range(200): 
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        episode_reward += reward

        if done:
            break
    
    # 使用 \r 实现单行动态刷新，看起来更酷
    print(f'\rEpisode: {i_episode+1}/{num_episodes}, Reward: {episode_reward:.2f}', end="")
    # 每20个回合，我们打印一个固定的日志并换行
    if (i_episode + 1) % 20 == 0:
        print("") # 换行

env.close()
print("\n训练完成!")