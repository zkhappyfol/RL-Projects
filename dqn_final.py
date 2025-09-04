import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Part 1: 环境设置 ---
env = gym.make("CartPole-v1")
# 【新增】设置设备，如果有GPU就用GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Part 2: 经验回放的记忆宫殿 ---
# 我们用 namedtuple 来存储每一条经验
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        # 使用deque作为高效的记忆存储器
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一条经验"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """随机抽取一批经验"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前记忆的数量"""
        return len(self.memory)

# --- Part 3: DQN 神经网络大脑 ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# ==========================================================
# ==============【新增】诊断函数 (1/2) =====================
# ==========================================================
def inspect_q_values(policy_net, memory, batch_size=128):
    """抽查并打印当前策略网络预测的平均最大Q值"""
    print("\n--- Q值检测 ---")
    if len(memory) < batch_size:
        print("经验池太小，无法抽样。")
        return

    experiences = memory.sample(batch_size)
    batch = Experience(*zip(*experiences))
    
    # 确保 next_state 不是 None
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    
    if len(non_final_next_states) > 0:
        with torch.no_grad():
            q_values = policy_net(non_final_next_states)
            avg_max_q = q_values.max(1)[0].mean().item()
            print(f"对 {batch_size} 个随机状态的平均最大Q值估算: {avg_max_q:.2f}")
    else:
        print("抽样中没有有效的 next_state。")
    print("---------------")

# ==========================================================
# ==============【新增】诊断函数 (2/2) =====================
# ==========================================================
def inspect_memory(memory, num_to_show=3):
    """抽查并打印 ReplayMemory 中的几条随机经验"""
    print("\n--- 记忆宫殿抽查 ---")
    print(f"当前容量: {len(memory)} / {memory.memory.maxlen}")
    
    if len(memory) == 0:
        print("记忆宫殿还是空的。")
        return

    num_to_show = min(num_to_show, len(memory))
    print(f"随机抽取 {num_to_show} 条经验:")
    
    random_sample = memory.sample(num_to_show)
    
    for i, exp in enumerate(random_sample):
        state_str = " ".join([f"{x:.2f}" for x in exp.state.squeeze().cpu().numpy()])
        action_str = exp.action.item()
        reward_str = exp.reward.item()
        
        print(f"  [样本 {i+1}]")
        print(f"    - 状态: [{state_str}] | 动作: {action_str} | 奖励: {reward_str}")
    print("--------------------")

# --- Part 4: 总指挥 Agent ---
class Agent:
    def __init__(self, n_observations, n_actions):
        self.n_actions = n_actions
        
        # 超参数
        self.BATCH_SIZE = 256

        self.GAMMA = 0.99
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.TAU = 0.005 # 目标网络软更新的系数
        self.LR = 1e-4

        # 创建主网络和目标网络
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 复制权重

        # 创建优化器和记忆宫殿
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(30000)
        
        self.steps_done = 0

        # 【新增】探索/利用计数器
        self.exploration_count = 0
        self.exploitation_count = 0


    def select_action(self, state):
        """根据 Epsilon-Greedy 策略选择动作"""
        sample = random.random()
        # 计算当前应该使用的 epsilon 值
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #print(f"steps_done: {self.steps_done}, sample的值: {sample:2f}, epsilon threshold的值: {eps_threshold:2f}")
        self.steps_done += 1

        if sample > eps_threshold:
            # 利用 (Exploitation)
            # 【修改】增加利用计数
            self.exploitation_count += 1
            with torch.no_grad():
                # .max(1) 返回每一行的最大值和其索引
                # 我们需要索引 [1]
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # 探索 (Exploration)
            # 【修改】增加探索计数
            self.exploration_count += 1
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    def optimize_model(self):
        """执行一步优化（学习）"""
        if len(self.memory) < self.BATCH_SIZE:
            #print("不学习")
            return # 如果记忆不够，就不学习

        # 1. 从记忆宫殿中采样
        experiences = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*experiences)) # 将一批经验转换成一个Experience元组

        # 2. 准备数据
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
        
        # 3. 计算预测Q值
        # policy_net计算出所有动作的Q值，然后我们用action_batch选出我们当初实际采取的动作的Q值
        predicted_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 4. 计算目标Q值
        """next_state_q_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            # 用“导师”target_net来计算新状态的Q值
            next_state_q_values = self.target_net(next_state_batch).max(1)[0]
        # 套用贝尔曼公式
        target_q_values = (next_state_q_values * self.GAMMA) + reward_batch"""
        # 4. 计算目标Q值
        # 首先，创建一个掩码(mask)来区分哪些是“结束状态”，哪些不是
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        # 找出所有不是“结束状态”的 next_state
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 创建一个和 BATCH_SIZE 一样长的零向量
        next_state_q_values = torch.zeros(self.BATCH_SIZE)

        # 只对那些不是“结束状态”的情况，用 target_net 计算它们的未来Q值
        with torch.no_grad():
            #next_state_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            # --- 【Double DQN 核心修改】 ---
            # 1. 让 policy_net（学生）来决定在 next_states 中，每个状态的最佳动作是什么
            best_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
            
            # 2. 让 target_net（教授）来为 policy_net 选出的那些最佳动作打分
            #    .gather(1, best_actions) 的作用就是从 target_net 的输出中，只挑出 best_actions 指定的那些动作的Q值
            next_state_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
            # --- 【修改结束】 ---

        # 套用贝尔曼公式，对于“结束状态”，它们的 next_state_q_values 保持为0
        target_q_values = (next_state_q_values * self.GAMMA) + reward_batch
        
        # 5. 计算损失
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))
        
        # 6. 反向传播，更新“学生”的权重
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 7. 软更新“导师”的权重
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

# --- Part 5: 主训练循环 ---
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = Agent(n_observations, n_actions)

num_episodes = 1000
episode_durations = []
rewards_in_period = 0 # 用于计算周期平均奖励
SHOW_EVERY = 50 # 每50个回合打印一次日志

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    total_reward = 0
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        
        reward = torch.tensor([reward])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # 存入记忆宫殿
        agent.memory.push(state, action, next_state, reward, done)

        state = next_state

        # 学习！
        agent.optimize_model()

        if done:
            episode_durations.append(t + 1)
            rewards_in_period += (t + 1)
            if (i_episode + 1) % SHOW_EVERY  == 0:
                average_reward = rewards_in_period / SHOW_EVERY
                print(f'\n==================== Episode {i_episode+1}/{num_episodes} ====================')
                print(f'最近{SHOW_EVERY}回合平均奖励: {average_reward:.2f}')
                
                # 打印探索/利用计数
                total_choices = agent.exploration_count + agent.exploitation_count
                if total_choices > 0:
                    exploit_pct = 100 * agent.exploitation_count / total_choices
                    print(f"探索/利用分析 (最近{SHOW_EVERY}回合): ")
                    print(f"  - 探索次数: {agent.exploration_count}")
                    print(f"  - 利用次数: {agent.exploitation_count} ({exploit_pct:.1f}%)")
                
                # 调用诊断函数
                inspect_q_values(agent.policy_net, agent.memory, agent.BATCH_SIZE)
                print("\n--- 记忆宫殿抽查 ---")
                print(f"当前容量: {len(agent.memory)} / {agent.memory.memory.maxlen}")
                #inspect_memory(agent.memory)

                # 重置计数器
                rewards_in_period = 0
                agent.exploration_count = 0
                agent.exploitation_count = 0
            break
            
print('训练完成!')
env.close()

# 简单的绘图来看训练效果
plt.figure(1)
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.plot(episode_durations)
plt.show()