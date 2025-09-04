import gymnasium as gym
import time
import ale_py

# 创建Pong游戏环境，并开启“人类”渲染模式
env = gym.make("ALE/Pong-v5", render_mode='human')

# 初始化环境
observation, info = env.reset()

print("="*40)
print("欢迎来到雅达利的世界！")
print(f"观察空间 (Observation Space) 的形状: {observation.shape}")
print(f"动作空间 (Action Space) 的大小: {env.action_space.n}")
print(f"动作的含义: {env.unwrapped.get_action_meanings()}")
print("="*40)

# 玩一个回合，看看是什么样
for _ in range(1000):
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 加一点延时，方便观察
    time.sleep(0.01)

    if done:
        print("一个回合结束！")
        observation, info = env.reset()
        time.sleep(1)

env.close()