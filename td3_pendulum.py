import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt # 引入绘图库
# vvvvvvvvvvvvvv  在这里添加下面两行 vvvvvvvvvvvvvv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#TD3 (Twin Delayed Deep Deterministic Policy Gradient)

# --- Part 1: 超参数设定 (无变化) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_CAPACITY = 10000 #记忆参数:决定了AI的“阅历”有多广，以及每次学习的“信息量”有多大
BATCH_SIZE = 128  #the same as memory_capacity
GAMMA = 0.99   #耐心参数:决定了AI是更看重眼前利益（低GAMMA）还是长远规划（高GAMMA）
TAU = 0.005    #学习转移参数:决定了“导师”网络向“学生”网络学习的速度，是“醍醐灌顶”还是“润物细无声”
ACTOR_LR = 1e-4    #学习率:这是最敏感、最核心的参数，决定了两个大脑的学习速度
CRITIC_LR = 1e-3   #通常评论家的学习率会比演员的高
NOISE_STDDEV = 0.1  #探索噪声 (Exploration Noise)
#TD3专属参数,用于稳定训练过程
POLICY_NOISE = 0.2          # 目标策略平滑的噪声标准差
NOISE_CLIP = 0.5            # 噪声的裁剪范围
POLICY_FREQ = 2             # 演员网络延迟更新的频率

def calculate_jitter_score(data_series):
    """
    计算一个时间序列的“颠簸度”分数。
    分数越低，代表曲线越平滑、噪声越少。
    """
    # 确保输入是 numpy 数组
    data_series = np.array(data_series)
    
    # 1. 使用 np.diff() 计算相邻数据点之间的差值
    differences = np.diff(data_series)
    
    # 2. 计算这些差值的标准差
    # 标准差越大，说明数据点之间的跳跃越剧烈、越不稳定
    jitter_score = np.std(differences)
    
    return jitter_score

# --- Part 2: 经验回放池 (无变化) ---
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

# --- Part 3 & 4: 网络结构 (无变化) ---
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
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Part 5: DDPG 总指挥 (修改train方法以返回loss) 改为TD3---
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # <--- 修改开始 --->
        # 创建两个评论家网络（双胞胎）
        self.critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2 = Critic(state_dim, action_dim).to(DEVICE)
        # 对应地，也创建两个目标评论家网络
        self.critic1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target = Critic(state_dim, action_dim).to(DEVICE)
        # 初始化权重
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 两个评论家需要一个共同的优化器
        # 我们把两个网络的参数都传给它
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=CRITIC_LR
        )
        # <--- 修改结束 --->

        self.memory = ReplayBuffer(MEMORY_CAPACITY)

        # <--- 新增开始 --->
        self.total_it = 0  # 训练迭代总次数计数器
        self.policy_noise = POLICY_NOISE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        # <--- 新增结束 --->

    def select_action(self, state, i_episode, num_episodes):# <--- 新增参数
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()

        # <--- 新增: 噪声衰减逻辑 --->
        # 设定初始和最终的噪声标准差
        start_noise_stddev = 0.2
        end_noise_stddev = 0.05
        # 线性衰减
        decay_rate = (start_noise_stddev - end_noise_stddev) / num_episodes
        current_noise_stddev = start_noise_stddev - (i_episode * decay_rate)
        # 确保噪声不会小于最终值
        current_noise_stddev = max(current_noise_stddev, end_noise_stddev)

        #noise = np.random.normal(0, self.max_action * NOISE_STDDEV, size=action.shape)
        # 使用动态计算出的噪声标准差
        noise = np.random.normal(0, self.max_action * current_noise_stddev, size=action.shape)
        return (action + noise).clip(-self.max_action, self.max_action)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None
        
        # 迭代计数器加一
        self.total_it += 1

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # --- 核心修改 1: 目标策略平滑 和 双评论家选择 ---
        with torch.no_grad():
            # 给目标演员的动作加上截断的高斯噪声
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)

            # 计算两个目标评论家网络的Q值
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            
            # 取两者中较小的值作为最终的目标Q值
            target_q = torch.min(target_q1, target_q2)
            
            # 贝尔曼方程
            target_q = rewards + (1 - dones) * GAMMA * target_q
        
        # --- 更新两个评论家网络 ---
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # 计算两个评论家的损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_loss_item = critic_loss.item()
        actor_loss_item = None # <--- 新增: 先假设actor没有更新
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 核心修改 2: 延迟更新演员网络和目标网络 ---
        # 每隔 policy_freq 次，才执行一次更新
        if self.total_it % self.policy_freq == 0:
            
            # --- 更新演员网络 ---
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            actor_loss_item = actor_loss.item() # <--- 新增: 如果更新了，就记录loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- 软更新所有四个目标网络 ---
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        # 为了绘图，我们还是返回loss，但在延迟更新时，actor_loss可能不存在
        # 为简化，我们暂时只返回critic_loss
        return actor_loss_item, critic_loss_item # <--- 修改: 返回两个loss变量

    def save_models(self, filename): # <--- 新增: 保存模型的方法
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{filename}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{filename}_critic2.pth")
        print(f"模型已保存到 {filename}_actor.pth 和 {filename}_critic1/2.pth")

# --- Part 6: 主训练循环 (修改以记录和绘图) ---
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)
num_episodes = 200

# <--- 新增: 用于记录数据的列表 ---
rewards = []
actor_losses = []
critic_losses = []

print("="*30 + "\n开始训练...\n" + "="*30)
for i_episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    temp_actor_loss, temp_critic_loss = [], []

    for t in range(200):
        action = agent.select_action(state, i_episode, num_episodes)# <--- 修改这里
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.memory.push(state, action, reward, next_state, done)
        
        # <--- 修改: 接收返回的loss值 ---
        a_loss, c_loss = agent.train()
        # 只要训练开始，critic_loss就一定有值
        if c_loss is not None:
            temp_critic_loss.append(c_loss)
        # 只有在actor更新时，a_loss才有值
        if a_loss is not None:
            temp_actor_loss.append(a_loss)

        state = next_state
        episode_reward += reward
        if done:
            break
            
    # <--- 新增: 记录每个回合的平均loss和总reward ---
    rewards.append(episode_reward)
    if temp_actor_loss:
        actor_losses.append(np.mean(temp_actor_loss))
        critic_losses.append(np.mean(temp_critic_loss))
    else: # 如果第一回合没开始训练，就记为0
        actor_losses.append(0)
        critic_losses.append(0)

    print(f'\rEpisode: {i_episode+1}/{num_episodes}, Reward: {episode_reward:.2f}', end="")
    if (i_episode + 1) % 20 == 0:
        print("")

env.close()
agent.save_models("td3_pendulum") # <--- 新增: 训练结束后保存模型
print("\n训练完成!")

# <--- 新增: 绘制多个图表 ---
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('TD3 训练过程分析')

# 绘制奖励曲线
axs[0].plot(rewards)
axs[0].set_title('每回合奖励 (Reward)')
axs[0].set_xlabel('回合 (Episode)')
axs[0].set_ylabel('总奖励 (Total Reward)')
jitter_score_reward = calculate_jitter_score(rewards)
print(f"td3 reward jitter: {jitter_score_reward:3f}")

# 绘制Actor Loss曲线
axs[1].plot(actor_losses)
axs[1].set_title('演员网络损失 (Actor Loss)')
axs[1].set_xlabel('回合 (Episode)')
axs[1].set_ylabel('损失 (Loss)')
jitter_score_act = calculate_jitter_score(actor_losses)
print(f"td3 actor jitter: {jitter_score_act:3f}")

# 绘制Critic Loss曲线
axs[2].plot(critic_losses)
axs[2].set_title('评论家网络损失 (Critic Loss)')
axs[2].set_xlabel('回合 (Episode)')
axs[2].set_ylabel('损失 (Loss)')
jitter_score_critic = calculate_jitter_score(critic_losses)
print(f"td3 critic jitter: {jitter_score_critic:3f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()