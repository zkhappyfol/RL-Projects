import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# vvvvvvvvvvvvvv  在这里添加下面两行 vvvvvvvvvvvvvv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- 1. 导入我们之前创建的世界模型结构 ---
# 注意：文件名需要匹配你保存世界模型的文件名
from world_model_frozenlake import WorldModel

# --- 2. 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "FrozenLake-v1"
STATE_DIM = 16  # 4x4 grid
ACTION_DIM = 4

ALPHA = 0.1  # Q-Learning的学习率
GAMMA = 0.99 # 折扣因子
# EPSILON = 0.1 # Epsilon-Greedy策略的探索率
EPS_START = 1.0  # 初始探索率
EPS_END = 0.01   # 最终探索率
EPS_DECAY = 1000 # 衰减速率
N_PLANNING_STEPS = 0 # 每次真实互动后，进行“脑内推演”的步数

# --- 3. Dyna-Q 智能体 ---
class DynaQAgent:
    def __init__(self, world_model_path):
        # 初始化一个空的Q-Table
        self.q_table = np.zeros((STATE_DIM, ACTION_DIM))
        
        # 初始化并加载我们训练好的世界模型
        self.world_model = WorldModel(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.world_model.load_state_dict(torch.load(world_model_path))
        self.world_model.eval() # 设置为评估模式
        print("世界模型加载成功！")
        
        # 记录所有经历过的(状态, 动作)对，用于规划
        self.observed_sa_pairs = set()

    def select_action(self, state, i_episode):# <--- 增加 i_episode 参数
        """Epsilon-Greedy策略"""
        """动态epsilon方法
        if random.random() < EPSILON:
            return random.randint(0, ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state, :])"""
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * i_episode / EPS_DECAY)
        
        if random.random() < eps_threshold:
            return random.randint(0, ACTION_DIM - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        """标准的Q-Learning更新规则"""
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def planning(self):
        """核心: 在“脑内”进行N次模拟和学习"""
        for _ in range(N_PLANNING_STEPS):
            # 随机选择一个过去经历过的(状态, 动作)对
            if not self.observed_sa_pairs: return # 如果还没经历过，就跳过
            
            s_rand, a_rand = random.choice(list(self.observed_sa_pairs))
            
            # --- 使用世界模型进行“想象” ---
            s_onehot = F.one_hot(torch.tensor([s_rand]), num_classes=STATE_DIM).float().to(DEVICE)
            a_onehot = F.one_hot(torch.tensor([a_rand]), num_classes=ACTION_DIM).float().to(DEVICE)
            
            with torch.no_grad():
                pred_ns_logits, pred_r, _ = self.world_model(s_onehot, a_onehot)
            
            # 解读想象的结果
            s_sim = torch.argmax(F.softmax(pred_ns_logits, dim=1)).item()
            r_sim = pred_r.item()
            
            # --- 用“想象”出的经验来更新Q-Table ---
            self.update_q_table(s_rand, a_rand, r_sim, s_sim)

# --- 4. 主训练循环 ---
print("开始Dyna-Q训练...")
env = gym.make(ENV_NAME, is_slippery=True)
agent = DynaQAgent("frozenlake_world_model.pth")

num_episodes = 2000
total_rewards = []

for i_episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, i_episode)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 1. 直接学习 (Direct RL)
        agent.update_q_table(state, action, reward, next_state)
        
        # 记录这次真实的经历
        agent.observed_sa_pairs.add((state, action))
        
        # 2. 规划 (Planning)
        agent.planning()
        
        state = next_state
        episode_reward += reward
        
    total_rewards.append(episode_reward)

    # <--- 新增的“工作汇报”模块 --->
    # 每100个回合，打印一次当前进度和最近的平均得分
    if (i_episode + 1) % 100 == 0:
        # 计算最近100个回合的平均奖励（即成功率）
        last_100_rewards_avg = np.mean(total_rewards[-100:])
        print(f"Episode {i_episode + 1}/{num_episodes} | 最近100回合成功率: {last_100_rewards_avg:.2f}")

print("训练完成!")
env.close()

# 绘图展示学习成果
plt.plot(total_rewards)
plt.title('Dyna-Q 学习曲线')
plt.xlabel('回合 (Episode)')
plt.ylabel('总奖励 (Total Reward)')
# 计算最近100个回合的平均奖励来观察收敛情况
avg_rewards = [np.mean(total_rewards[i:i+100]) for i in range(len(total_rewards) - 100)]
plt.plot(np.arange(100, len(total_rewards)), avg_rewards, label='最近100回合平均奖励')
plt.legend()
plt.show()