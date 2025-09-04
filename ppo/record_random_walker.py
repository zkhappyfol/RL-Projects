import gymnasium as gym
import time
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# --- 配置 ---
ENV_ID = "BipedalWalker-v3"
VIDEO_FOLDER = "./random_walker_videos/" # 存放到一个新文件夹，避免覆盖
VIDEO_LENGTH = 2000 # 一个回合最长1600步，确保能录完
NUM_EPISODES_TO_RECORD = 3 # 我们想录制3个回合

# --- 1. 创建并包装环境 ---
# 我们只创建这一个环境，它从诞生之初就具备录像功能
env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

env = VecVideoRecorder(env, VIDEO_FOLDER,
                       # 这个lambda函数的意思是：在每个回合开始时（step=0），都触发一次录像
                       record_video_trigger=lambda step: step == 0, 
                       video_length=VIDEO_LENGTH,
                       name_prefix=f"random-walker-{ENV_ID}")

# --- 2. 开始随机表演和录制 ---
print("开始录制随机行动的机器人...")
obs = env.reset()

for i_episode in range(NUM_EPISODES_TO_RECORD):
    done = False
    episode_reward = 0
    t = 0
    while not done:
        # 从环境中随机采样一个动作
        action = [env.action_space.sample()] # VecEnv需要一个列表或Numpy数组形式的动作
        
        obs, reward, done, info = env.step(action)
        
        # VecEnv的reward和done是数组，我们需要取出里面的值
        episode_reward += reward[0]
        t += 1

    print(f"回合 {i_episode+1} 结束 | 步数: {t}, 总奖励: {episode_reward:.2f}")

# 关闭环境，这时VecVideoRecorder会自动完成视频的保存
env.close()
print(f"\n表演结束！ {NUM_EPISODES_TO_RECORD}个回合的视频已保存到 '{VIDEO_FOLDER}' 文件夹下。")