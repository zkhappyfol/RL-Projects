import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
import time

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# 将Gymnasium的日志级别设置为ERROR
# 这会隐藏掉所有级别低于ERROR的日志，比如WARNING和INFO
gym.logger.set_level(gym.logger.ERROR)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- Part 1: 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e6)  # 记忆宫殿容量
BATCH_SIZE = 1024       # 学习批量大小
GAMMA = 0.99            # 折扣因子
TAU = 1e-3              # 目标网络软更新系数
LR_ACTOR = 1e-4         # 演员学习率
LR_CRITIC = 1e-3        # 评论家学习率
NOISE_STDDEV = 0.1      # 动作噪声

# --- Part 2: 网络结构 (我们之前定义好的) ---
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)
    def forward(self, obs):
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        action = torch.sigmoid(self.layer3(x)) # 输出范围 0 到 1
        return action

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(total_obs_dim + total_action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Part 3: 记忆宫殿 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)
    def __len__(self):
        return len(self.memory)

# --- Part 4: MADDPG 指挥中心 (完整版) ---
class MADDPGAgent:
    def __init__(self, env):
        self.env = env
        #self.agents = env.agents
        self.agents = env.possible_agents
        self.n_agents = len(self.agents)
        
        self.obs_dims = [self.env.observation_space(agent).shape[0] for agent in self.agents]
        self.action_dims = [self.env.action_space(agent).shape[0] for agent in self.agents]
        total_obs_dim = sum(self.obs_dims)
        total_action_dim = sum(self.action_dims)

        # 创建演员和评论家网络 (包括目标网络)
        self.actors = [Actor(obs_dim, act_dim).to(DEVICE) for obs_dim, act_dim in zip(self.obs_dims, self.action_dims)]
        self.critics = [Critic(total_obs_dim, total_action_dim).to(DEVICE) for _ in self.agents]
        self.actor_targets = [Actor(obs_dim, act_dim).to(DEVICE) for obs_dim, act_dim in zip(self.obs_dims, self.action_dims)]
        self.critic_targets = [Critic(total_obs_dim, total_action_dim).to(DEVICE) for _ in self.agents]
        
        # 复制权重
        for actor, target in zip(self.actors, self.actor_targets):
            target.load_state_dict(actor.state_dict())
        for critic, target in zip(self.critics, self.critic_targets):
            target.load_state_dict(critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critics]

        self.memory = ReplayBuffer(BUFFER_SIZE)

    def select_actions(self, observations):
        actions = {}
        with torch.no_grad():
            for i, agent_id in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(observations[agent_id]).to(DEVICE)
                action = self.actors[i](obs_tensor).cpu().numpy()
                # 添加探索噪声
                noise = np.random.normal(0, NOISE_STDDEV, size=self.action_dims[i])
                actions[agent_id] = (action + noise).clip(0, 1) # 裁剪到有效范围
        return actions

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # 1. 从记忆宫殿采样
        obs_n, actions_n, rewards_n, next_obs_n, dones_n = self.memory.sample(BATCH_SIZE)

        # --- 修正开始: 重新整理数据，确保形状正确 ---
        # 目标: 将离散的智能体数据，整合成一个大的“团队”Tensor
        # all_obs 的目标形状是 [BATCH_SIZE, total_obs_dim]
        
        # 将每个时间步的所有智能体观察拼接在一起
        all_obs = torch.FloatTensor([np.concatenate(obs) for obs in obs_n]).to(DEVICE)
        all_actions = torch.FloatTensor([np.concatenate(action) for action in actions_n]).to(DEVICE)
        all_next_obs = torch.FloatTensor([np.concatenate(next_obs) for next_obs in next_obs_n]).to(DEVICE)
        # --- 修正结束 ---

        # --- 更新所有评论家网络 ---
        #all_obs = torch.FloatTensor(np.concatenate(obs_n, axis=1)).to(DEVICE)
        #all_next_obs = torch.FloatTensor(np.concatenate(next_obs_n, axis=1)).to(DEVICE)
        
        with torch.no_grad():
            # 计算所有智能体的下一个动作 (来自目标演员网络)
            all_next_actions = torch.cat([self.actor_targets[i](torch.FloatTensor(next_obs_n[i]).to(DEVICE)) for i in range(self.n_agents)], dim=1)
            # 计算目标Q值
            target_q_values = self.critic_targets[0](all_next_obs, all_next_actions) # 只用一个评论家来评估全局Q值
            rewards = torch.FloatTensor(rewards_n[0]).unsqueeze(1).to(DEVICE) # 假设所有智能体奖励相同
            dones = torch.FloatTensor(dones_n[0]).unsqueeze(1).to(DEVICE)
            target_q = rewards + GAMMA * (1 - dones) * target_q_values

        for i in range(self.n_agents):
            current_actions = torch.FloatTensor(np.concatenate(actions_n, axis=1)).to(DEVICE)
            current_q = self.critics[i](all_obs, current_actions)
            
            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # --- 更新所有演员网络 ---
        for i in range(self.n_agents):
            obs_i = torch.FloatTensor(obs_n[i]).to(DEVICE)
            pred_actions = list(torch.FloatTensor(actions_n[j]).to(DEVICE) for j in range(self.n_agents))
            pred_actions[i] = self.actors[i](obs_i)
            pred_actions_cat = torch.cat(pred_actions, dim=1)
            
            actor_loss = -self.critics[i](all_obs, pred_actions_cat).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # --- 软更新所有目标网络 ---
        for actor, target in zip(self.actors, self.actor_targets):
            for param, target_param in zip(actor.parameters(), target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# --- Part 5: 主训练循环 ---
if __name__ == '__main__':
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True, render_mode="rgb_array")
    
    agent_group = MADDPGAgent(env)
    
    print("="*40)
    print("开始MADDPG团队集训...")
    print("="*40)

    episode_rewards = []
    for i_episode in range(1, 2001):
        observations, infos = env.reset()
        episode_reward = 0
        
        while env.agents:
            actions = agent_group.select_actions(observations)
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 将字典形式的数据转换为列表形式，方便存储
            obs_list = [observations[agent] for agent in env.agents]
            action_list = [actions[agent] for agent in env.agents]
            reward_list = [rewards[agent] for agent in env.agents]
            next_obs_list = [next_observations[agent] for agent in env.agents]
            done_list = [terminations[agent] or truncations[agent] for agent in env.agents]
            
            agent_group.memory.push((obs_list, action_list, reward_list, next_obs_list, done_list))
            
            agent_group.update()
            
            observations = next_observations
            episode_reward += sum(reward_list)

        episode_rewards.append(episode_reward)
        if i_episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {i_episode}, 最近100回合平均奖励: {avg_reward:.2f}")

    env.close()