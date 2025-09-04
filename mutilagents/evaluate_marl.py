import torch
import numpy as np
from pettingzoo.mpe import simple_spread_v3
import time
from moviepy import ImageSequenceClip
import os

# --- 关键：我们需要从训练脚本中，原封不动地复制Actor类的“设计图纸” ---
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        )
    def forward(self, obs): return self.net(obs)

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PREFIX = "maddpg_spread" # <--- 必须和训练时保存的前缀一致
VIDEO_FOLDER = "./marl_videos_trained/"
VIDEO_FILENAME = "maddpg_spread_performance.mp4"
NUM_EPISODES_TO_RECORD = 3
FPS = 15

# --- 1. 创建环境和加载模型 ---
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=100, continuous_actions=True, render_mode="rgb_array")
agents = env.possible_agents
obs_dims = [env.observation_space(agent).shape[0] for agent in agents]
action_dims = [env.action_space(agent).shape[0] for agent in agents]
n_agents = len(agents)

# 创建空的演员列表
actors = [Actor(obs_dim, act_dim).to(DEVICE) for obs_dim, act_dim in zip(obs_dims, action_dims)]

# 加载每个演员的权重
for i, actor in enumerate(actors):
    try:
        actor.load_state_dict(torch.load(f"{MODEL_PREFIX}_actor_{i}.pth"))
        actor.eval() # 设置为评估模式
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{MODEL_PREFIX}_actor_{i}.pth'。请先完成训练并保存模型。")
        exit()
print("所有冠军大脑加载成功！")

# --- 2. 开始录制巅峰对决 ---
all_frames = []
print(f"开始录制 {NUM_EPISODES_TO_RECORD} 场团队表演...")

for i_episode in range(NUM_EPISODES_TO_RECORD):
    observations, infos = env.reset()
    # 拍摄第一帧
    frame = env.render()
    all_frames.append(frame)
    
    while env.agents:
        with torch.no_grad():
            # 为每个智能体分别决策
            actions = {}
            for i, agent_id in enumerate(agents):
                obs_tensor = torch.FloatTensor(observations[agent_id]).to(DEVICE)
                # 使用对应的演员网络，不加噪声
                action = actors[i](obs_tensor).cpu().numpy()
                actions[agent_id] = action
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 拍摄新一帧
        frame = env.render()
        all_frames.append(frame)
        
    print(f"回合 {i_episode+1} 录制结束。")

env.close()

# --- 3. 合成视频 ---
print("\n开始合成视频...")
os.makedirs(VIDEO_FOLDER, exist_ok=True)
video_path = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)
try:
    clip = ImageSequenceClip(all_frames, fps=FPS)
    clip.write_videofile(video_path, logger=None)
    print(f"视频成功保存到: {video_path}")
except Exception as e:
    print(f"视频合成失败，错误: {e}")