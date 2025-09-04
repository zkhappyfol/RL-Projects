import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# --- 配置 ---
ENV_ID = "BipedalWalker-v3"
MODEL_PATH = "./best_model/best_model.zip"  # 我们训练好的模型
VIDEO_FOLDER = "./videos/" # 存放视频的文件夹
VIDEO_LENGTH = 1000 # 录制多长

# --- 1. 创建用于评估的环境 ---
# 我们需要用DummyVecEnv把环境包一下，这是SB3录像工具的要求
eval_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

# --- 2. 用VecVideoRecorder包装环境 ---
# 这个包装器会自动把环境的运行过程录制成视频
eval_env = VecVideoRecorder(eval_env, VIDEO_FOLDER,
                            record_video_trigger=lambda step: step == 0, 
                            video_length=VIDEO_LENGTH,
                            name_prefix=f"ppo-{ENV_ID}")

# --- 3. 加载训练好的模型 ---
try:
    model = PPO.load(MODEL_PATH, env=eval_env)
    print("模型加载成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 '{MODEL_PATH}'。请确保训练脚本已成功保存模型。")
    exit()

# --- 4. 开始表演和录像 ---
print("开始录制表演视频...")
obs = eval_env.reset()
for _ in range(VIDEO_LENGTH + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if done.any():
        break

# 关闭环境，录像会自动保存
eval_env.close()
print(f"表演结束！视频已保存到 '{VIDEO_FOLDER}' 文件夹下。")