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
NOISE_STDDEV = 0.05  #探索噪声 (Exploration Noise)
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
# --- Part 3: SAC 演员网络 (概率性) ---
from torch.distributions import Normal # <--- 引入正态分布

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        
        # 新增: 两个输出头，一个用于均值(mean)，一个用于标准差(log_std)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        # 计算均值和log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # 限制 log_std 的范围，防止标准差变得过大或过小，增加训练稳定性
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std

    def sample(self, state):
        # 1. 从前向传播中获取分布参数
        mean, log_std = self.forward(state)
        
        # 2. 计算标准差
        std = torch.exp(log_std)
        
        # 3. 创建正态分布对象
        normal = Normal(mean, std)
        
        # 4. 从分布中采样一个动作 (rsample支持反向传播)
        x_t = normal.rsample()
        
        # 5. 使用 tanh 函数对动作进行“挤压”，使其范围在 [-1, 1] 之间
        y_t = torch.tanh(x_t)
        
        # 6. 将动作缩放到环境的实际范围 [-max_action, max_action]
        action = y_t * self.max_action
        
        # 7. 计算动作的对数概率 (log_prob)，这是SAC损失函数的核心
        log_prob = normal.log_prob(x_t)
        
        # 8. 对log_prob进行修正，因为tanh改变了概率分布
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

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

# --- Part 5: SAC 总指挥 (修改train方法以返回loss) ---
# 建议将 class TD3Agent 改名为 class SACAgent
class SACAgent:
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

        # <--- 新增: 熵温度 alpha 的自动调节装置 --->
        # 目标熵，通常设置为-action_dim，这是一个经验法则
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).to(DEVICE)
        
        # 我们不直接优化alpha，而是优化log_alpha，让alpha恒为正，且优化过程更稳定
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        # <--- 新增开始 --->
        #self.total_it = 0  # 训练迭代总次数计数器
        #self.policy_noise = POLICY_NOISE
        #self.noise_clip = NOISE_CLIP
        #self.policy_freq = POLICY_FREQ
        # <--- 新增结束 --->

    def select_action(self, state, evaluate=False):
        """根据策略选择动作，区分训练和评估"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        # evaluate标志用于区分是训练还是测试
        if evaluate is False:
            # 训练时，我们从分布中“随机采样”一个动作，天生就带探索性
            action, _ = self.actor.sample(state)
        else:
            # 测试时，我们选择分布的“均值”作为最有可能的最优动作
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean) * self.max_action

        return action.cpu().data.numpy().flatten()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None, None # 返回 actor_loss, critic_loss, alpha

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # --- 更新评论家网络 (Critic Update) ---
        with torch.no_grad():
            # 1. 从actor网络获取下一个动作及其log_prob
            next_actions, next_log_prob = self.actor.sample(next_states)
            
            # 2. 计算目标Q值
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            min_q_target = torch.min(target_q1, target_q2)
            
            # 3. 引入熵项！这就是SAC的核心
            alpha = torch.exp(self.log_alpha) # 获取当前的alpha值
            target_q = rewards + (1 - dones) * GAMMA * (min_q_target - alpha * next_log_prob)

        # 计算当前Q值和评论家损失
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化评论家
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新演员网络和Alpha (Actor and Alpha Update) ---
        # SAC通常会同步更新演员和Alpha，不再需要延迟更新
        
        # 1. 计算演员损失
        actions_pred, log_prob_pred = self.actor.sample(states)
        q_value_pred = torch.min(self.critic1(states, actions_pred), self.critic2(states, actions_pred))
        
        actor_loss = (alpha.detach() * log_prob_pred - q_value_pred).mean()
        
        # 2. 优化演员
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. 计算Alpha的损失
        alpha_loss = -(self.log_alpha * (log_prob_pred + self.target_entropy).detach()).mean()
        
        # 4. 优化Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 软更新目标网络 ---
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return actor_loss.item(), critic_loss.item(), torch.exp(self.log_alpha).item()

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

# 在主程序部分，我们正式将Agent改名为SACAgent
agent = SACAgent(state_dim, action_dim, max_action)
num_episodes = 200

# <--- 新增/修改: 用于记录最终图表数据的四个列表 ---
episode_rewards = []
avg_actor_losses = []
avg_critic_losses = []
avg_alphas = []

print("="*30 + "\n开始训练...\n" + "="*30)
for i_episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    # <--- 新增: 用于记录当前回合内的临时数据 ---
    temp_actor_losses, temp_critic_losses, temp_alphas = [], [], []
    #temp_actor_loss, temp_critic_loss = [], []

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.memory.push(state, action, reward, next_state, done)
        
        # <--- 修改: 接收返回的loss值 ---
        #a_loss, c_loss = agent.train()
        # 修改train的调用，接收三个返回值
        a_loss, c_loss, alpha_val = agent.train() 

        # 只要训练开始，critic_loss就一定有值
        if c_loss is not None:
            temp_critic_losses.append(c_loss)
        # 只有在actor更新时，a_loss才有值
        if a_loss is not None:
            temp_actor_losses.append(a_loss)
        # 在记录平均loss的地方，也记录alpha的平均值
        if alpha_val is not None:
            temp_alphas.append(alpha_val) # <--- 新增

        state = next_state
        episode_reward += reward
        if done:
            break


        # <--- 新增/修改: 回合结束后，计算平均值并存入总账本 ---
    episode_rewards.append(episode_reward)
    avg_critic_losses.append(np.mean(temp_critic_losses) if temp_critic_losses else 0)
    avg_alphas.append(np.mean(temp_alphas) if temp_alphas else 0)
    avg_actor_losses.append(np.mean(temp_actor_losses) if temp_actor_losses else 0)

    print(f'\rEpisode: {i_episode+1}/{num_episodes}, Reward: {episode_reward:.2f}', end="")
    if (i_episode + 1) % 20 == 0:
        print("")

env.close()
agent.save_models("sac_pendulum") # <--- 新增: 训练结束后保存模型
print("\n训练完成!")

# <--- 新增: 绘制多个图表 ---
fig, axs = plt.subplots(4, 1, figsize=(10, 18))
fig.suptitle('SAC 训练过程分析')

# 绘制奖励曲线
axs[0].plot(episode_rewards)
axs[0].set_title('每回合奖励 (Reward)')
axs[0].set_xlabel('回合 (Episode)')
axs[0].set_ylabel('总奖励 (Total Reward)')
jitter_score_reward = calculate_jitter_score(episode_rewards)
print(f"sac reward jitter: {jitter_score_reward:3f}")

# 绘制Actor Loss曲线
axs[1].plot(avg_actor_losses)
axs[1].set_title('演员网络损失 (Actor Loss)')
axs[1].set_xlabel('回合 (Episode)')
axs[1].set_ylabel('损失 (Loss)')
jitter_score_act = calculate_jitter_score(avg_actor_losses)
print(f"sac actor jitter: {jitter_score_act:3f}")


# 绘制Critic Loss曲线
axs[2].plot(avg_critic_losses)
axs[2].set_title('评论家网络损失 (Critic Loss)')
axs[2].set_xlabel('回合 (Episode)')
axs[2].set_ylabel('损失 (Loss)')
jitter_score_critic = calculate_jitter_score(avg_critic_losses)
print(f"sac critic jitter: {jitter_score_critic:3f}")


# 绘制alphas曲线
axs[3].plot(avg_alphas)
axs[3].set_title('熵温度 Alpha 的变化 (alphas)')
axs[3].set_xlabel('回合 (Episode)')
axs[3].set_ylabel('Alpha 值 (Value)')
jitter_score_alphas = calculate_jitter_score(avg_alphas)
print(f"sac alphas jitter: {jitter_score_alphas:3f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()