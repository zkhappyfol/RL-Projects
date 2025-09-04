from pettingzoo.mpe import simple_spread_v3
import time

# 1. 创建环境
# N=3 代表有3个智能体, local_ratio=0.5 控制奖励函数, max_cycles=25 是一个回合的最大步数
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, render_mode="human")

# 2. 初始化环境
# 在多智能体环境中，reset()会返回一个字典，键是每个智能体的名字
observations, infos = env.reset()

print("="*40)
print("欢迎来到多智能体的世界！")
print(f"智能体列表: {env.agents}")
# 打印出第一个智能体的观察空间和动作空间
agent_0 = env.agents[0]
print(f"'{agent_0}'的观察空间: {env.observation_space(agent_0)}")
print(f"'{agent_0}'的动作空间: {env.action_space(agent_0)}")
print("="*40)

# 3. 运行主循环
for _ in range(3): # 玩3个回合
    while env.agents: # 只要环境里还有智能体
        
        # 为每个智能体随机选择一个动作
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # 执行所有动作
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        time.sleep(0.1) # 延时，方便观察
        
    print("一个回合结束！")
    time.sleep(1)
    observations, infos = env.reset()

env.close()