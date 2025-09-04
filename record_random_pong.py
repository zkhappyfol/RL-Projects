import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np
from collections import deque
import ale_py

# -------------------------------------------------------------
# 关键：这个脚本也需要视觉预处理，所以我们把两个包装器类
# 从训练脚本中原封不动地复制到这里，让脚本“自给自足”
# -------------------------------------------------------------
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
            low=0, high=255, shape=(k, height, width), dtype=np.uint8
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
        frames_np = np.concatenate(list(self.frames), axis=2)
        return frames_np.transpose(2, 0, 1)

# -------------------------------------------------------------

# --- 配置 ---
VIDEO_FOLDER = "./pong_videos_random/" # 使用一个新文件夹，避免覆盖
NUM_EPISODES_TO_RECORD = 3 # 录制3个回合的“菜鸟”比赛

# --- 1. 创建并封装环境 ---
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
# 套上和训练时完全一样的预处理流程
env = PreprocessFrame(env, width=84, height=84)
env = StackFrames(env, k=4)
# 套上录像机
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda episode_id: True,
    name_prefix="random-pong-match"
)

# --- 2. 开始随机表演与录制 ---
print(f"开始录制 {NUM_EPISODES_TO_RECORD} 场‘菜鸟’AI的比赛...")
for i in range(NUM_EPISODES_TO_RECORD):
    state, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # 【核心区别】在这里，我们使用随机动作，而不是加载模型
        action = env.action_space.sample() 

        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = observation
        episode_reward += reward

    print(f"第 {i+1} 场菜鸟赛结束, 得分: {episode_reward}")

env.close()
print(f"\n录制完成！视频已保存到 '{VIDEO_FOLDER}' 文件夹。")