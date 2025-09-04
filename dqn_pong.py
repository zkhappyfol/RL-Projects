import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
import cv2
import matplotlib.pyplot as plt
import time
import math
import ale_py

# --- Part 1: 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
GAMMA = 0.99
# Epsilon-Greedy 参数
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 30000
# 软更新目标网络参数
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 100000 # 记忆宫殿容量
NUM_EPISODES = 1000 # 训练回合数

# --- Part 2: 预处理包装器 (我们之前写好的) ---
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )
    def observation(self, obs):
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(gray_obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized_obs[:, :, None]

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        height, width, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, height, width), dtype=np.uint8 # <--- 注意: 改为通道在前
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    def observation(self, obs):
        self.frames.append(obs)
        return self._get_obs()
    def _get_obs(self):
        # 将deque中的k个帧在通道维度上拼接
        frames_np = np.concatenate(list(self.frames), axis=2)
        # 从 (H, W, C) 转换到 (C, H, W)
        return frames_np.transpose(2, 0, 1)

# --- Part 3: CNN 大脑 (我们之前写好的) ---
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def _get_conv_out(self, x):
        o = self.conv_layers(x)
        return int(torch.flatten(o, 1).size(1))
    def forward(self, x):
        # 输入图像的像素值范围是[0, 255]，我们需要将其归一化到[0, 1]
        x = x.float() / 255.0
        conv_out = self.conv_layers(x)
        flattened = torch.flatten(conv_out, 1)
        q_values = self.fc_layers(flattened)
        return q_values

# --- Part 4: 记忆宫殿 (老朋友) ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Experience(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# --- Part 5: 主训练逻辑 ---
def main():
    # 1. 创建并封装环境
    env = gym.make("ALE/Pong-v5")
    env = PreprocessFrame(env, width=84, height=84)
    env = StackFrames(env, k=4)

    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = state.shape

    # 2. 创建网络、优化器、记忆宫殿
    policy_net = CnnDQN(n_observations, n_actions).to(DEVICE)
    target_net = CnnDQN(n_observations, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)
    
    steps_done = 0
    episode_rewards = []

    highest_score = -float('inf') # <--- 新增：初始化最高分记录为负无穷大

    print("="*40)
    print(f"开始在 {DEVICE} 上训练 Pong...")
    print(f"总共进行 {NUM_EPISODES} 个回合的训练。")
    print("="*40)
    
    start_time = time.time() # <--- 新增: 记录开始时间

    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.uint8, device=DEVICE).unsqueeze(0)
        
        episode_reward = 0
        for t in count():
            # 3. 选择动作 (Epsilon-Greedy)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)

            # <--- 新增: “心跳”日志 --- >
            if steps_done % 1000 == 0:
                elapsed_time = time.time() - start_time
                print(f"  ... 总步数: {steps_done}, Epsilon: {eps_threshold:.4f}, 已用时: {elapsed_time:.2f}s")


            # 4. 与环境互动
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                next_state = torch.tensor(np.array(observation), dtype=torch.uint8, device=DEVICE).unsqueeze(0)

            # 5. 存入记忆
            memory.push(state, action, reward, next_state, done)
            state = next_state

            # 6. 学习！(执行一步优化)
            if len(memory) > BATCH_SIZE:
                experiences = memory.sample(BATCH_SIZE)
                batch = Experience(*zip(*experiences))
                
                # ... (省略了处理 non_final_next_states 的代码，这部分与我们TD3中的修复类似) ...
                # 为了简洁，我们直接使用PyTorch官方教程的简化写法
                
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                predicted_q_values = policy_net(state_batch).gather(1, action_batch)
                
                next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                
                target_q_values = (next_state_values * GAMMA) + reward_batch

                criterion = nn.SmoothL1Loss()
                loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # 梯度裁剪
                optimizer.step()

            # 7. 软更新目标网络
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(episode_reward)
                # <--- 修改: 让回合结束的日志更详细 --- >
                avg_reward_last_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"回合 {i_episode+1} 结束 | 步数: {t+1} | 得分: {episode_reward} | 最近100回合平均分: {avg_reward_last_100:.2f}")
                
                # <--- 新增的“里程碑日志” --->
                if episode_reward > highest_score:
                    highest_score = episode_reward
                    # 打印一条带 🎉 庆祝表情的、特别显眼的新纪录信息！
                    print(f"🎉 新纪录诞生！ Highest Score: {highest_score:.2f}")
                # <--- 新增结束 --->

                break
    
    print("训练完成!")

    # <--- 在这里添加下面的代码 --->
    # 1. 定义保存路径
    MODEL_SAVE_PATH = "dqn_pong_model.pth"
    # 2. 保存 policy_net 的“大脑”参数
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已成功保存到: {MODEL_SAVE_PATH}")
    # <--- 添加结束 --->

    # 绘图
    plt.plot(episode_rewards)
    plt.title('DQN on Pong Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# --- Part 6: 启动！ ---
if __name__ == '__main__':
    main()