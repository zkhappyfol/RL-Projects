import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
# <--- 新增: 导入回调函数 --->
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. 创建环境 ---
# BipedalWalker-v3
env_id = "BipedalWalker-v3"
# SB3推荐使用矢量化环境，可以同时跑多个环境加速训练，我们先用1个
env = make_vec_env(env_id, n_envs=4) # <--- 优化: 同时跑4个环境，加速数据收集

# --- 2. 设置“自动存档”的回调函数 ---
# 我们创建一个独立的评估环境
eval_env = gym.make(env_id)
# EvalCallback会每隔5000步，在评估环境上测试一次模型
# 然后把历史最佳模型，保存在 'best_model' 文件夹下
eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

# --- 2. 创建PPO模型 ---
# 我们直接从SB3库里，实例化一个PPO智能体
# "MlpPolicy" 指使用我们之前学过的多层感知机(全连接网络)作为策略网络
# verbose=1 会让它在训练时打印详细的日志
model = PPO("MlpPolicy", env, verbose=1)

# --- 3. 启动训练 ---
# 这一行代码，就代替了我们之前写的整个复杂的训练循环！
# 我们让它训练20万个时间步
#model.learn(total_timesteps=200000)
#print("开始使用 Stable Baselines3 进行训练...")

# 我们将训练量提升到一百万步
TOTAL_TIMESTEPS = 1000000 
print(f"开始进行 {TOTAL_TIMESTEPS} 步的深度训练...")
# 在.learn()中加入callback
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

print("训练完成！")
print(f"最佳模型已保存在 './best_model/' 文件夹下的 best_model.zip 文件中。")

"""在evaluate里面看表演
# --- 4. 保存并测试训练好的模型 ---
model.save("ppo_bipedalwalker")
print("模型已保存到 ppo_bipedalwalker.zip")

# 加载模型并观看表演
del model # 删除旧模型
model = PPO.load("ppo_bipedalwalker")

print("\n开始观看训练成果...")

# <--- 修改开始 --->
# VecEnv的reset只返回obs，所以我们只用一个变量来接收
obs = env.reset()
# <--- 修改结束 --->
# obs, info = env.reset()

for _ in range(3000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # if done:
        # obs, info = env.reset()
    # VecEnv会在一个回合结束后自动重置，我们可以根据done标志来判断
    # done在这里是一个数组，比如[True]或[False]
    if done.any():
        print("一个回合结束，机器人将自动重置...")
        time.sleep(2) # 暂停2秒，方便我们看清楚
"""

env.close()
eval_env.close()