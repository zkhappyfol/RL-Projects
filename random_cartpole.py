import gymnasium as gym
import time
import math

# 1. 创建 CartPole 环境，render_mode="human" 可以让我们看到图形界面
env = gym.make("CartPole-v1", render_mode="human")

# 2. 初始化环境，获取第一个观察值 (observation)
# observation 包含了4个数字：[小车位置, 小车速度, 杆子角度, 杆子角速度]
observation, info = env.reset()
total_step = 0
total_reward = 0

# 循环玩1000个时间步
for _ in range(1000):
    # 3. 随机选择一个动作 (0: 向左推, 1: 向右推)
    action = env.action_space.sample() 

    # 4. 执行动作，获取环境反馈
    # observation: 新的状态
    # reward: 这个动作获得的奖励 (在CartPole里，只要没倒就是+1)
    # terminated: 游戏是否因为达到目标或失败而结束 (比如杆子倒了),这个应该是12度就认为会倒
    # truncated: 游戏是否因为达到时间限制等非失败原因而结束
    # info: 额外信息 (我们暂时不用管)
    observation, reward, terminated, truncated, info = env.step(action)

    # 从 observation 中提取杆子的角度（它是第3个元素，索引为2）
    pole_angle_radians = observation[2]
    
    # 将弧度转换为角度
    pole_angle_degrees = math.degrees(pole_angle_radians)

    pole_location = observation[0]

    # 打印出每一步的信息，方便我们观察
    print(f"action: {action}, pole_location: {pole_location:.3f}, angle: {pole_angle_degrees:.2f}, Reward: {reward}")
    total_step = total_step + 1
    total_reward = total_reward + 1

    # 如果游戏结束了 (杆子倒了)，就重置游戏
    if terminated or truncated:
        print(f"--- Episode Finished, total steps: {total_step}, total rewards: {total_reward} ---")
        observation, info = env.reset()
        total_step = 0
        total_reward = 0
        # 在重置后稍微暂停一下，让我们能看清楚
        time.sleep(1)

# 5. 关闭环境，释放资源
env.close()