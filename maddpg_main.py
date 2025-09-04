import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
import time
import gymnasium as gym


# --- Part 1: 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 1024
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
NOISE_STDDEV = 0.1
UPDATE_EVERY = 2 # 每隔几步学习一次

# --- Part 2: 网络结构 (无变化) ---
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Sigmoid() # 使用Sigmoid确保输出在[0,1]
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=1)
        return self.net(x)

# --- Part 3: 记忆宫殿 (无变化) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # 将数据按类型分开
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.memory)

# --- Part 4: MADDPG 指挥中心 (逻辑重构) ---
class MADDPGAgent:
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        self.n_agents = len(self.agents)
        
        self.obs_dims = [self.env.observation_space(agent).shape[0] for agent in self.agents]
        self.action_dims = [self.env.action_space(agent).shape[0] for agent in self.agents]
        total_obs_dim = sum(self.obs_dims)
        total_action_dim = sum(self.action_dims)

        self.actors = [Actor(obs_dim, act_dim).to(DEVICE) for obs_dim, act_dim in zip(self.obs_dims, self.action_dims)]
        self.critics = [Critic(total_obs_dim, total_action_dim).to(DEVICE) for _ in self.agents]
        self.actor_targets = [Actor(obs_dim, act_dim).to(DEVICE) for obs_dim, act_dim in zip(self.obs_dims, self.action_dims)]
        self.critic_targets = [Critic(total_obs_dim, total_action_dim).to(DEVICE) for _ in self.agents]
        
        self._sync_targets()

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critics]

        self.memory = ReplayBuffer(BUFFER_SIZE)

    def select_actions(self, observations):
        actions = {}
        pure_actions = {} # <--- 新增: 存储不加噪声的动作
        with torch.no_grad():
            for i, agent_id in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(observations[agent_id]).to(DEVICE)

                # 演员网络输出的是“纯粹的”决策
                pure_action = self.actors[i](obs_tensor).cpu().numpy()
                pure_actions[agent_id] = pure_action
                #action = self.actors[i](obs_tensor).cpu().numpy()
                noise = np.random.normal(0, NOISE_STDDEV, size=self.action_dims[i])
                # 先做裁剪和加噪声
                #noisy_action = (action + noise).clip(0, 1)
                # 最终执行的是加了噪声并裁剪的动作
                final_action = (pure_action + noise).clip(0, 1)
                actions[agent_id] = final_action.astype(np.float32)
                
                #actions[agent_id] = (action + noise).clip(0, 1)

                # 【关键修正】在最后，将数据类型强制转换为float32
                #actions[agent_id] = noisy_action.astype(np.float32)
        return actions, pure_actions 

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # --- 数据整理 ---
        all_states = torch.FloatTensor([np.concatenate(s) for s in states]).to(DEVICE)
        all_actions = torch.FloatTensor([np.concatenate(a) for a in actions]).to(DEVICE)
        all_next_states = torch.FloatTensor([np.concatenate(ns) for ns in next_states]).to(DEVICE)
        
        rewards = torch.FloatTensor([r[0] for r in rewards]).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor([d[0] for d in dones]).unsqueeze(1).to(DEVICE)
        
        # --- 更新评论家 ---
        with torch.no_grad():
            next_states_T = list(zip(*next_states))
            all_next_actions = torch.cat([
                self.actor_targets[i](torch.FloatTensor(next_states_T[i]).to(DEVICE))
                for i in range(self.n_agents)
            ], dim=1)
            
            target_q = self.critic_targets[0](all_next_states, all_next_actions)
            target_q = rewards + GAMMA * (1 - dones) * target_q

        for i in range(self.n_agents):
            current_q = self.critics[i](all_states, all_actions)
            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # --- 更新演员 ---
        for i in range(self.n_agents):
            states_i = torch.FloatTensor(list(zip(*states))[i]).to(DEVICE)
            
            pred_actions = [self.actors[j](torch.FloatTensor(list(zip(*states))[j]).to(DEVICE)) for j in range(self.n_agents)]
            pred_actions_cat = torch.cat(pred_actions, dim=1)
            
            actor_loss = -self.critics[i](all_states, pred_actions_cat).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        
        self._soft_update_targets()

    def _sync_targets(self):
        for actor, target in zip(self.actors, self.actor_targets):
            target.load_state_dict(actor.state_dict())
        for critic, target in zip(self.critics, self.critic_targets):
            target.load_state_dict(critic.state_dict())
            
    def _soft_update_targets(self):
        for actor, target in zip(self.actors, self.actor_targets):
            for param, target_param in zip(actor.parameters(), target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                # <--- 在类的末尾，添加下面这个新方法 --->
    def save_models(self, prefix="maddpg_model"):
        """保存所有演员网络的权重"""
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{prefix}_actor_{i}.pth")
        print(f"\n所有演员网络已保存，文件名前缀为: {prefix}")

# --- Part 5: 主训练循环 (逻辑重构) ---
if __name__ == '__main__':
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    agent_group = MADDPGAgent(env)
    
    #print("="*40 + "\n开始MADDPG团队集训 (稳健版)...\n" + "="*40)
    print("="*40 + "\n开始MADDPGAgent团队集训 (详细日志版)...\n" + "="*40)
    
    episode_rewards = []
    total_steps = 0
    
    for i_episode in range(1, 5001):
        observations, infos = env.reset()
        episode_reward = 0
        
        # PettingZoo v1.25.0, a round is not limited by steps but by cycles now
        # We need a step counter for our learning trigger
        for step in range(100): # max_cycles = 100
            total_steps += 1
            
            # 当回合刚开始或结束后，env.agents可能是空的，需要处理
            if not env.agents:
                break

            # <--- 修改: 接收两个动作字典 --->
            final_actions, pure_actions = agent_group.select_actions(observations)

            
            # actions = agent_group.select_actions(observations)
            next_observations, rewards, terminations, truncations, infos = env.step(final_actions)
            
            # 将字典数据转换为列表
            obs_list = [observations[agent] for agent in agent_group.agents]
            action_list = [final_actions[agent] for agent in agent_group.agents]
            reward_list = [rewards[agent] for agent in agent_group.agents]
            next_obs_list = [next_observations[agent] for agent in agent_group.agents]
            done_list = [terminations[agent] or truncations[agent] for agent in agent_group.agents]
            
            agent_group.memory.push((obs_list, action_list, reward_list, next_obs_list, done_list))
            
            # 每隔几步学习一次，而不是每一步都学
            if total_steps % UPDATE_EVERY == 0:
                agent_group.update()
            
            observations = next_observations
            episode_reward += sum(reward_list)

        episode_rewards.append(episode_reward)
        # <--- 修改: 打印更丰富的日志 --->
        if i_episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            # 打印记忆宫殿的大小
            memory_size = len(agent_group.memory)
            
            # 随机选一个智能体（比如第一个），看看它的纯动作和最终动作的差异
            agent_id_to_check = agent_group.agents[0]
            action_diff = np.linalg.norm(final_actions[agent_id_to_check] - pure_actions[agent_id_to_check])

            print(f"Episode: {i_episode}, Avg Reward: {avg_reward:.2f}, " \
                  f"Memory Size: {memory_size}/{BUFFER_SIZE}, " \
                  f"Action Noise Mag: {action_diff:.4f}")
            #print(f"Episode: {i_episode}, 最近100回合平均奖励: {avg_reward:.2f}")

    # <--- 在 env.close() 前面，添加下面这行代码 --->
    agent_group.save_models("maddpg_spread") # 我们可以给模型起个名字

    env.close()