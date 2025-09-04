import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. PPO "大脑" 的基本骨架 ---

# Actor-Critic共享网络的一部分，用于提取特征
class ActorCriticShared(nn.Module):
    def __init__(self, state_dim):
        super(ActorCriticShared, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        return F.relu(self.layer2(x))

# 演员网络 (驾驶员)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.shared = ActorCriticShared(state_dim)
        # 演员的“头部”，输出动作的均值
        self.mean_head = nn.Linear(256, action_dim) 
        # 我们需要一个变量来控制探索的随机程度(标准差)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        action_mean = torch.tanh(self.mean_head(features)) # 用tanh将动作均值压缩到[-1, 1]
        return action_mean

# 评论家网络 (导航员)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.shared = ActorCriticShared(state_dim)
        # 评论家的“头部”，输出对状态的价值评估
        self.value_head = nn.Linear(256, 1)

    def forward(self, state):
        features = self.shared(state)
        return self.value_head(features)

# --- 2. 侦察行动 ---
if __name__ == '__main__':
    # 创建BipedalWalker环境
    # render_mode='human' 让我们能看到画面
    env = gym.make('BipedalWalker-v3', render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("="*40)
    print("Bipedal Walker 环境侦察")
    print(f"状态空间维度 (State Dim): {state_dim}")
    print(f"动作空间维度 (Action Dim): {action_dim}")
    print(f"最大动作值 (Max Action): {env.action_space.high[0]}")
    print("="*40)

    # 创建一个傻瓜机器人，让它随机行动
    for i_episode in range(3):
        state, info = env.reset()
        episode_reward = 0
        done = False
        t = 0
        while not done:
            # 随机选择一个在动作范围内的动作
            action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            t += 1
            
            if done:
                print(f"回合 {i_episode+1} 结束 | 步数: {t}, 总奖励: {episode_reward:.2f}")
                break
    
    env.close()