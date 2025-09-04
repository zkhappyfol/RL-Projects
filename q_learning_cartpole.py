import gymnasium as gym
import numpy as np
import math
import time
import pickle
import os

#加载Q Table
def load_q_table(filename="q_table.pkl"):
    """从文件加载 Q-Table, 如果文件不存在则创建一个新的"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        print(f"Q-Table loaded from {filename}")
        return q_table
    else:
        print(f"No Q-Table found, creating a new one.")
        return {} # 返回一个空字典作为新的 Q-Table
        
# 在你的程序一开始，就这样创建 Q-Table
# my_q_table = load_q_table()

# ==================================
# Part 1: 环境和“大脑”的初始化
# ==================================

# 1. 加载环境
# 在训练阶段，我们不需要图形化界面，所以注释掉 render_mode
# 这样能让训练速度快上几百倍！
env = gym.make("CartPole-v1")

# 2. 定义状态空间离散化的“桶” (Buckets)
# 状态有4个值：[小车位置, 小车速度, 杆子角度, 杆子角速度]
# 我们把每个值的范围切分成几个“桶”
# 杆子角度对成功至关重要，所以我们给它分的桶最多（6个）
# buckets = (1, 1, 6, 3) # (位置, 速度, 角度, 角速度)
buckets = (10, 10, 12, 12) # (位置, 速度, 角度, 角速度)

# 3. 创建 Q-Table (我们的小抄)
# 表格的维度 = (1, 1, 6, 3) 再加上 动作空间的大小 (2, 即向左或向右)
# 所以 Q-Table 的形状是 [1, 1, 6, 3, 2]
# np.zeros 会创建一个所有元素都为0的数组
# q_table = np.zeros(buckets + (env.action_space.n,))
"""q_table = {
    (1, 1, 5, 3): [10.5, 12.1],  # 状态 (4, 7, 7, 5) 下, 动作0的Q值是10.5, 动作1是12.1
    (1, 1, 4, 2): [11.2, 9.8],
    ... # 更多状态
}"""
#用字典方式来创建q_table
q_table = load_q_table()
print(f"加载文件后, Q-Table 的大小是: {len(q_table)}")

# 4. 定义学习相关的超参数 (Hyperparameters)
# 这些参数就像是调节机器人学习方式的旋钮
LEARNING_RATE = 0.1         # 学习率 α: 对新知识的吸收程度
DISCOUNT = 0.99             # 折扣因子 γ: 对未来奖励的重视程度
EPISODES = 25000            # 训练回合数: 让机器人练习多少次
SHOW_EVERY = 500           # 每隔多少回合，打印一次训练进度

# 探索率 ε (Epsilon) 的设置
# 我们希望机器人在一开始多“探索”，后期多“利用”学到的知识
epsilon = 1.0               # 初始探索率
EPSILON_DECAY = 0.99998       # 探索率衰减值, 每次都乘以这个数
MIN_EPSILON = 0.01 # 即使在最后，也保持 1% 的探索率

# ==================================
# Part 2: 状态离散化函数
# ==================================

# 这是一个关键函数，负责把连续的状态值 (比如-0.023) 转换成离散的“桶”的索引 (比如 2)
def discretize_state(state):
    # 定义每个状态值的合理范围 (根据官方文档和经验)
    #env.observation_space.high[0]=2.4, low[0]=-2.4, high[2]=12度,low[2]=-12度
    #速度范围是[-0.5,0.5], 角速度范围是[-50度, 50度]
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    
    # 计算每个桶的大小, 在这个维度上(lower-upper)是x%的位置上
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    
    # 将状态值映射到对应的桶索引, round四舍五入
    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    
    # 确保索引在有效范围内 [0, bucket_size - 1]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    
    return tuple(new_state)

#保存Q Table
def save_q_table(q_table, filename="q_table.pkl"):
    """将 Q-Table 保存到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-Table saved to {filename}")

# 在你的训练循环结束后，或者每隔一定回合数，调用这个函数
# save_q_table(my_q_table)



# 查看Q Table
def inspect_q_table(q_table, num_entries=5):
    """打印 Q-Table 的前几项，看看它学到了什么"""
    print("\n--- Inspecting Q-Table ---")
    if not np.any(q_table):
        print("Q-Table is empty.")
        return
        
    for i, (state, q_values) in enumerate(q_table.items()):
        if i >= num_entries:
            break
        print(f"State: {state}, Q-Values (Left, Right): {q_values}")
    print(f"Total entries in Q-Table: {len(q_table)}")
    print("--------------------------\n")

# 在训练过程中，可以偶尔调用这个函数来观察学习进展
# inspect_q_table(my_q_table)



# ==================================
# Part 3: 主训练循环
# ==================================

print("开始训练...")

# --- 在你的主 for 循环开始之前，初始化一个变量 ---
# 用来累积一个周期内的总奖励
rewards_in_period = 0

for episode in range(EPISODES):
    # 每个回合开始时，重置环境
    current_state, info = env.reset()
    current_discrete_state = discretize_state(current_state)
    
    done = False
    total_reward = 0

    exploi = 0
    explor = 0

    while not done:
        # 1. 首先，确保当前状态在Q-Table中有对应的条目
        #  如果这是第一次遇到这个状态，就为它创建一个全零的Q值列表
        if current_discrete_state not in q_table:
            q_table[current_discrete_state] = np.zeros(env.action_space.n) # env.action_space.n 通常是 2

        # Epsilon-Greedy 策略：在探索和利用之间做选择
        if np.random.random() > epsilon:
            # 利用 (Exploitation): 从Q-Table中选择最优动作
            action = np.argmax(q_table[current_discrete_state])
            exploi += 1
        else:
            # 探索 (Exploration): 随机选择动作
            action = env.action_space.sample()
            explor += 1
            
        # 执行动作，获取环境反馈
        new_state, reward, terminated, truncated, info = env.step(action)
        # === 新增的奖励塑造规则 ===
        # 如果小车跑偏了，就给它一点点惩罚
        # new_state[0] 代表小车的位置，abs()是取绝对值
        # 离中心越远，惩罚越大
        #reward = reward - abs(new_state[0]) * 0.05
        new_discrete_state = discretize_state(new_state)
        
        done = terminated or truncated
        total_reward += reward

        # ==========================================================
        # ============== 奖励塑造 (REWARD SHAPING) ==================
        # ==========================================================
        # 我们定义一个用于更新Q值的新奖励变量
        reward_for_update = reward # 正常情况下，它就是环境给的 +1

        # 如果回合因为失败而结束 (而不是因为成功达到最大步数)
        if done and total_reward < 499: # 这里的499是一个参考，确保不是成功通关
            reward_for_update = -20  # 给予一个巨大的负奖励作为惩罚！
        # ==========================================================

        # 2. 同样，也为新状态做一样的检查
        if new_discrete_state not in q_table:
            q_table[new_discrete_state] = np.zeros(env.action_space.n)

        # 如果游戏没有结束，就更新 Q-Table
        if not done:
            # Q-Learning 的核心更新公式

            # current_q = q_table[current_discrete_state + (action,)]
            current_q = q_table[current_discrete_state][action]
            max_future_q = np.max(q_table[new_discrete_state])
            
            # THE FORMULA! 贝尔曼函数
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            
            #q_table[current_discrete_state + (action,)] = new_q
            q_table[current_discrete_state][action] = new_q

        # 如果游戏是因为失败而结束的，我们也应该更新最后一步的Q值
        elif done:
            # 对于导致失败的那个动作，它的Q值就直接是那个巨大的惩罚
            q_table[current_discrete_state][action] = reward_for_update
                    
        current_discrete_state = new_discrete_state

    # 衰减 Epsilon，让机器人越来越相信自己的判断
    if epsilon > MIN_EPSILON: # 保持一定的探索率
        epsilon *= EPSILON_DECAY

    # 一个回合结束后，将这个回合的奖励（步数）累加到周期总奖励中
    rewards_in_period += total_reward

    # 打印训练进度
    if (episode + 1) % SHOW_EVERY == 0:
        # 计算这个周期（SHOW_EVERY个回合）的平均奖励
        average_reward = rewards_in_period / SHOW_EVERY
        print(f"回合: {episode + 1}, 平均奖励: {average_reward:.2f}")
        print(f"当前内存中的Q Table的大小: {len(q_table)}")
        print(f"当前 Epsilon: {epsilon:.5f}") # 观察 Epsilon 的值
        print(f"利用次数: {exploi}, 探索次数: {explor}")
        save_q_table(q_table,"q_table.pkl")
        # 【关键】打印完后，将周期总奖励清零，为下一个周期做准备
        rewards_in_period = 0


print("训练完成！")
inspect_q_table(q_table)

# ==================================
# Part 4: 展示训练成果 (下一阶段任务)
# ==================================
# 训练完成后，我们可以用学好的 Q-Table 来看看机器人的表现

# 先关闭用于训练的环境
env.close()

# 创建一个带图形界面的新环境
env_test = gym.make("CartPole-v1", render_mode="human")
state, info = env_test.reset()
done = False
total_step = 0

print("\n开始展示训练成果...")


while not done:
    # 在测试阶段，我们只“利用”，不“探索”
    total_step += 1
    discrete_state = discretize_state(state)
    action = np.argmax(q_table[discrete_state])
    
    # 执行动作
    state, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated
    
    # 短暂暂停，方便肉眼观察
    time.sleep(0.5)

print("展示结束。")
print(f"--- Episode Finished, total steps: {total_step} ---")
env_test.close()