import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import ale_py

# vvvvvvvvvvvvvv  在这里添加下面两行 vvvvvvvvvvvvvv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- 包装器 1: 图像预处理 ---
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
        # 定义处理后的观察空间形状
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # 1. 转为灰度图
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # 2. 缩小尺寸
        resized_obs = cv2.resize(gray_obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # 3. 增加一个通道维度，从 (H, W) 变成 (H, W, 1)
        return resized_obs[:, :, None]

# --- 包装器 2: 堆叠帧 ---
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k # 要堆叠的帧数
        # 使用deque来高效地存储最近的k个帧
        self.frames = deque([], maxlen=k)
        
        # 定义堆叠后的观察空间形状
        height, width, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 重置时，用第一帧填充整个deque
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        # 将deque中的k个帧，在最后一个维度(通道)上拼接起来
        return np.concatenate(list(self.frames), axis=2)

# --- 测试我们的包装器 ---
if __name__ == '__main__':
    # 1. 创建原始环境
    env_raw = gym.make("ALE/Pong-v5")
    obs_raw_shape = env_raw.observation_space.shape
    print(f"原始状态形状: {obs_raw_shape}")
    
    # 2. 套上第一个滤镜：图像预处理
    # 我们将图像处理成 84x84 的灰度图
    env_preprocessed = PreprocessFrame(env_raw, width=84, height=84)
    obs_preprocessed_shape = env_preprocessed.observation_space.shape
    print(f"图像预处理后形状: {obs_preprocessed_shape}")

    # 3. 套上第二个滤镜：堆叠4帧
    env_stacked = StackFrames(env_preprocessed, k=4)
    obs_stacked_shape = env_stacked.observation_space.shape
    print(f"堆叠4帧后形状: {obs_stacked_shape}")
    
    # 4. 看看实际输出
    obs, info = env_stacked.reset()
    print(f"\n经过包装器处理后，reset()返回的实际状态形状: {obs.shape}")
    
    # 我们可以用matplotlib看看这些处理后的帧是什么样子
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    fig.suptitle('堆叠的4帧灰度图 (一个完整的状态)')
    for i in range(4):
        # obs的形状是 (H, W, C)，所以我们取第i个通道
        axes[i].imshow(obs[:, :, i], cmap='gray')
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    plt.show()