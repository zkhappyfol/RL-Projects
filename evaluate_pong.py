import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import time
import ale_py

# ===================================================================
#  第一部分：将所有需要的“设计图纸”直接复制到这里
# ===================================================================

# --- 包装器 1: 图像预处理 ---
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

# --- 包装器 2: 堆叠帧 ---
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

# --- CNN大脑 "设计图纸" ---
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
        x = x.float() / 255.0
        conv_out = self.conv_layers(x)
        flattened = torch.flatten(conv_out, 1)
        q_values = self.fc_layers(flattened)
        return q_values

# ===================================================================
#  第二部分：主程序 - 表演与录制
# ===================================================================

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./best_model/dqn_pong_model.pth" # 要加载的模型文件
VIDEO_FOLDER = "./pong_videos/"   # 存放视频的文件夹
NUM_EPISODES_TO_RECORD = 5      # 我们想录制5个回合的比赛

# --- 1. 创建并封装用于评估的环境 ---
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = PreprocessFrame(env, width=84, height=84)
env = StackFrames(env, k=4)
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda episode_id: True,
    name_prefix="dqn-pong-match"
)

# --- 2. 加载“冠军大脑” ---
n_actions = env.action_space.n
# 在包装后的环境上调用reset，以获得正确的初始状态形状
state, info = env.reset()
n_observations = state.shape

policy_net = CnnDQN(n_observations, n_actions).to(DEVICE)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()
print("冠军大脑加载成功！")

# --- 3. 开始巅峰对决与录制 ---
print(f"开始录制 {NUM_EPISODES_TO_RECORD} 场巅峰对决...")
for i in range(NUM_EPISODES_TO_RECORD):
    # reset() 已经在循环外调用过一次，对于录像包装器，
    # 我们只需要在循环内处理 done 的情况
    done = False
    episode_reward = 0
    
    while not done:
        state = torch.tensor(np.array(state), dtype=torch.uint8, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        state = observation
        episode_reward += reward

    print(f"第 {i+1} 场比赛结束, 得分: {episode_reward}")
    # 录像包装器会自动处理回合结束后的reset

env.close()
print(f"\n录制完成！视频已保存到 '{VIDEO_FOLDER}' 文件夹。")