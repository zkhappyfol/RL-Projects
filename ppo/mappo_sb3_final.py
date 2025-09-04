import gymnasium as gym
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder, DummyVecEnv
import supersuit as ss
import os
from moviepy.editor import ImageSequenceClip
import numpy as np
import warnings

# --- 0. 环境准备 ---
# 忽略所有恼人的警告
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import logging
logging.basicConfig(level=logging.ERROR)


# ===================================================================
# Part 1: 训练阶段
# ===================================================================

# --- 1. 创建 PettingZoo 环境 ---
env_raw = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=100, continuous_actions=True)

# --- 2. 使用 Supersuit 包装器，将其转换为与SB3兼容的格式 ---
# 这是实现“参数共享”的关键，所有智能体将由同一个PPO模型控制
env = ss.pettingzoo_env_to_vec_env_v1(env_raw)
env = VecMonitor(env)

# --- 3. 设置回调函数，自动保存最佳模型 ---
# 创建一个独立的、未被包装的评估环境
eval_env_raw = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=100, continuous_actions=True)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env_raw)
eval_env = VecMonitor(eval_env)

BEST_MODEL_SAVE_PATH = "./best_mappo_model/"
LOG_DIR = "./mappo_logs/"
eval_callback = EvalCallback(eval_env, best_model_save_path=BEST_MODEL_SAVE_PATH,
                             log_path=LOG_DIR, eval_freq=10000,
                             deterministic=True, render=False)

# --- 4. 创建并训练PPO模型 (MAPPO) ---
TOTAL_TIMESTEPS = 1000000 # 训练一百万步
MODEL_SAVE_PATH = os.path.join(BEST_MODEL_SAVE_PATH, "best_model.zip") # 最佳模型的最终路径

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

print("="*40)
print(f"开始使用 MAPPO (Stable Baselines3 PPO) 进行最终的团队集训...")
print(f"将进行 {TOTAL_TIMESTEPS} 步的训练，最佳模型将被自动保存。")
print("="*40)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

print("\n训练完成！")
env.close()
eval_env.close()

# ===================================================================
# Part 2: 评估与录像阶段
# ===================================================================

print("\n" + "="*40)
print("开始加载最佳模型并录制表演视频...")
print("="*40)

# --- 1. 加载我们训练好的最佳模型 ---
try:
    eval_model = PPO.load(MODEL_SAVE_PATH)
except FileNotFoundError:
    print(f"错误: 找不到模型文件 '{MODEL_SAVE_PATH}'。")
    exit()

# --- 2. 创建一个纯净的、原始的PettingZoo环境用于录像 ---
video_env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=100, continuous_actions=True, render_mode="rgb_array")

# --- 3. 手动录制循环 ---
all_frames = []
NUM_EPISODES_TO_RECORD = 3
FPS = 15

for i_episode in range(NUM_EPISODES_TO_RECORD):
    observations, infos = video_env.reset()
    episode_reward = 0
    
    frame = video_env.render()
    all_frames.append(frame)
    
    while video_env.agents:
        # 将观察数据从字典转换为SB3模型需要的Numpy数组
        obs_array = np.array([observations[agent] for agent in video_env.agents])
        
        # 使用加载的模型来预测所有智能体的动作
        action, _ = eval_model.predict(obs_array, deterministic=True)
        
        # 将SB3输出的数组动作，转换回PettingZoo需要的字典格式
        actions_dict = {agent: act for agent, act in zip(video_env.agents, action)}

        observations, rewards, terminations, truncations, infos = video_env.step(actions_dict)
        
        frame = video_env.render()
        all_frames.append(frame)
        
        episode_reward += sum(rewards.values())

    print(f"录制的回合 {i_episode+1} 结束, 总奖励: {episode_reward:.2f}")

video_env.close()

# --- 4. 手动合成视频 ---
print("\n开始合成视频...")
VIDEO_FOLDER = "./marl_videos_final/"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
video_filename = "mappo_spread_final_performance.mp4"
video_path = os.path.join(VIDEO_FOLDER, video_filename)
try:
    clip = ImageSequenceClip(all_frames, fps=FPS)
    clip.write_videofile(video_path, logger='bar')
    print(f"视频成功保存到: {video_path}")
except Exception as e:
    print(f"视频合成失败，错误: {e}")