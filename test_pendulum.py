import gymnasium as gym
import torch
import time
from ddpg_pendulum2 import Actor # <--- 从我们刚才的文件中导入Actor网络结构

# --- 参数设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILENAME = "ddpg_pendulum_best" # <--- 要加载的模型文件名

SEED = 42 # 选择一个你喜欢的数字

# --- 加载环境和模型 ---
# render_mode='human' 让我们能看到图形界面
env = gym.make('Pendulum-v1', render_mode='human') 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 创建一个和训练时结构完全一样的Actor网络

actor = Actor(state_dim, action_dim, max_action).to(DEVICE)

# 加载我们保存好的权重
try:
    actor.load_state_dict(torch.load(f"{MODEL_FILENAME}_actor.pth"))
    print("Actor模型权重加载成功！")
except FileNotFoundError:
    print(f"错误：找不到模型文件 '{MODEL_FILENAME}_actor.pth'。请先运行训练脚本。")
    exit()

actor.eval() # 设置为评估模式（这对于有Dropout等层的网络很重要）

# --- 运行并观看 ---
state, info = env.reset(seed=SEED)
done = False
total_reward = 0

print("\n开始表演... (按 Ctrl+C 停止)")
try:
    while not done:
        # env.render() # 在新版Gymnasium中，render_mode='human'会自动渲染
        
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        # 在测试时，我们只“利用”，不加噪声
        with torch.no_grad():
            action = actor(state_tensor).cpu().data.numpy().flatten()
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        total_reward += reward
        
        # 加一点延时，不然速度太快看不清
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n表演被用户中断。")

print(f"表演结束。最终奖励: {total_reward:.2f}")
env.close()