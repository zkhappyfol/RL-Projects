import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt # 引入绘图库
# vvvvvvvvvvvvvv  在这里添加下面两行 vvvvvvvvvvvvvv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SEED = 42 # 选择一个你喜欢的数字
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Part 1: 超参数设定 (无变化) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
NOISE_STDDEV = 0.1

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

# --- Part 5: DDPG 总指挥 (修改train方法以返回loss) ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * NOISE_STDDEV, size=action.shape)
        return (action + noise).clip(-self.max_action, self.max_action)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None # <--- 修改: 如果不训练，返回None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * GAMMA * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return actor_loss.item(), critic_loss.item() # <--- 新增: 返回loss值

    def save_models(self, filename): # <--- 新增: 保存模型的方法
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")
        print(f"模型已保存到 {filename}_actor.pth 和 {filename}_critic.pth")


# ==========================================================
# ============【关键修改】把所有执行代码放入这里 ============
# ==========================================================
if __name__ == '__main__':

    # --- Part 6: 主训练循环 (修改以记录和绘图) ---
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dim, action_dim, max_action)
    num_episodes = 200

    # 【新增】用于追踪最佳表现的变量，初始设为一个极小的值
    best_reward = -float('inf') 

    # <--- 新增: 用于记录数据的列表 ---
    rewards = []
    actor_losses = []
    critic_losses = []

    print("="*30 + "\n开始训练...\n" + "="*30)
    for i_episode in range(num_episodes):
        state, info = env.reset(seed=SEED)
        episode_reward = 0
        temp_actor_loss, temp_critic_loss = [], []

        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            
            # <--- 修改: 接收返回的loss值 ---
            a_loss, c_loss = agent.train()
            if a_loss is not None:
                temp_actor_loss.append(a_loss)
                temp_critic_loss.append(c_loss)

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

        # ==========================================================
        # ============【关键修改】模型检查点逻辑 =====================
        # ==========================================================
        # 如果当前回合的奖励，超过了我们记录的历史最佳奖励
        if episode_reward > best_reward:
            best_reward = episode_reward # 更新最佳奖励
            agent.save_models("ddpg_pendulum_best") # 保存当前表现最好的模型！
            print(f"\n🎉 新纪录！发现在第 {i_episode+1} 回合，最佳奖励: {best_reward:.2f}。模型已保存！")
        # ==========================================================

    # (可选) 你仍然可以在最后保存一个最终模型，用于对比
    agent.save_models("ddpg_pendulum_final")


    # env.close()
    # agent.save_models("ddpg_pendulum") # <--- 新增: 训练结束后保存模型
    print("\n训练完成!")

    # ... 你的训练循环 ...

    #env.close() # <--- 把这行先注释掉或者移到最后

    # ==========================================================
    # ============【新增】在训练脚本末尾进行即时验证 ============
    # ==========================================================
    print("\n--- 训练结束，开始即时验证保存的模型 ---")

    # 创建一个新的 Actor 网络实例
    validation_actor = Actor(state_dim, action_dim, max_action).to(DEVICE)

    # 加载刚刚保存的权重
    try:
        validation_actor.load_state_dict(torch.load("ddpg_pendulum_best_actor.pth"))
        print("模型权重加载成功，开始进行可视化测试...")
    except FileNotFoundError:
        print("错误：找不到刚刚保存的模型文件。")
        exit()

    validation_actor.eval()

    # 在一个新的、带图形界面的环境中进行测试
    test_env = gym.make('Pendulum-v1', render_mode='human')
    state, info = test_env.reset(seed=SEED)
    total_reward = 0
    done = False

    try:
        step_count = 0 
        while not done:
            # 【关键修改】在循环的开始，强制渲染一帧画面
            test_env.render() 
            step_count += 1
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
            with torch.no_grad():
                action = validation_actor(state_tensor).cpu().data.numpy().flatten()
            
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            time.sleep(0.05) # 短暂延时方便观察
        print(f"验证循环总共运行了 {step_count} 步。")

    except KeyboardInterrupt:
        print("\n用户中断验证。")

    print(f"\n即时验证结束。最终奖励: {total_reward:.2f}")
    test_env.close()
    env.close() # <--- 移到这里

    # <--- 新增: 绘制多个图表 ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('DDPG 训练过程分析')

    # 绘制奖励曲线
    axs[0].plot(rewards)
    axs[0].set_title('每回合奖励 (Reward)')
    axs[0].set_xlabel('回合 (Episode)')
    axs[0].set_ylabel('总奖励 (Total Reward)')
    jitter_score_reward = calculate_jitter_score(rewards)
    print(f"ddpg reward jitter: {jitter_score_reward:3f}")

    # 绘制Actor Loss曲线
    axs[1].plot(actor_losses)
    axs[1].set_title('演员网络损失 (Actor Loss)')
    axs[1].set_xlabel('回合 (Episode)')
    axs[1].set_ylabel('损失 (Loss)')
    jitter_score_act = calculate_jitter_score(actor_losses)
    print(f"ddpg actor jitter: {jitter_score_act:3f}")

    # 绘制Critic Loss曲线
    axs[2].plot(critic_losses)
    axs[2].set_title('评论家网络损失 (Critic Loss)')
    axs[2].set_xlabel('回合 (Episode)')
    axs[2].set_ylabel('损失 (Loss)')
    jitter_score_critic = calculate_jitter_score(critic_losses)
    print(f"ddpg critic jitter: {jitter_score_critic:3f}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()