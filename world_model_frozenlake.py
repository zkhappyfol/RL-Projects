import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # <--- 添加这一行
import numpy as np
import random
from collections import deque

# --- 1. 超参数和环境设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "FrozenLake-v1"
env = gym.make(ENV_NAME, is_slippery=True)

STATE_DIM = env.observation_space.n # 状态空间大小 (16)
ACTION_DIM = env.action_space.n # 动作空间大小 (4)

# --- 2. 世界模型 (未来预测器) ---
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(WorldModel, self).__init__()
        # 输入维度 = 状态的独热编码 + 动作的独热编码
        input_dim = state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 三个输出头
        self.next_state_head = nn.Linear(128, state_dim)
        self.reward_head = nn.Linear(128, 1)
        self.done_head = nn.Linear(128, 1)

    def forward(self, state_onehot, action_onehot):
        # 拼接输入
        x = torch.cat([state_onehot, action_onehot], dim=1)
        x = self.net(x)
        
        # 预测三个结果
        pred_next_state_logits = self.next_state_head(x)
        pred_reward = self.reward_head(x)
        pred_done_logits = self.done_head(x)
        
        return pred_next_state_logits, pred_reward, pred_done_logits

# --- 3. 数据收集和准备 ---
def collect_random_data(env, num_steps=1000):
    """在环境中随机行动，收集经验"""
    memory = []
    state, info = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        if done:
            state, info = env.reset()
        else:
            state = next_state
    return memory

def prepare_batch(batch, state_dim, action_dim):
    """将一批经验数据转换成可用于训练的Tensor"""
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # 独热编码
    states_onehot = F.one_hot(torch.tensor(states, dtype=torch.int64), num_classes=state_dim).float()
    actions_onehot = F.one_hot(torch.tensor(actions, dtype=torch.int64), num_classes=action_dim).float()
    
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.int64) # 交叉熵损失需要LongTensor
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    return states_onehot, actions_onehot, rewards, next_states, dones

if __name__ == '__main__':

    # --- 4. 训练循环 ---
    print("开始构建世界模型...")
    # 1. 创建模型和优化器
    model = WorldModel(STATE_DIM, ACTION_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 2. 定义三种不同的损失函数，对应三个预测任务
    loss_fn_next_state = nn.CrossEntropyLoss()
    loss_fn_reward = nn.MSELoss()
    loss_fn_done = nn.BCEWithLogitsLoss() # 自带Sigmoid

    # 3. 收集数据
    print("正在收集中... (随机行动10000步)")
    memory = collect_random_data(env, num_steps=10000)
    print(f"收集到 {len(memory)} 条经验。")

    # 4. 开始训练
    print("开始训练模型...")
    epochs = 50
    batch_size = 64
    for epoch in range(epochs):
        random.shuffle(memory)
        epoch_losses = []
        for i in range(0, len(memory), batch_size):
            batch = memory[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            states_onehot, actions_onehot, rewards, next_states, dones = prepare_batch(batch, STATE_DIM, ACTION_DIM)
            states_onehot, actions_onehot = states_onehot.to(DEVICE), actions_onehot.to(DEVICE)
            rewards, next_states, dones = rewards.to(DEVICE), next_states.to(DEVICE), dones.to(DEVICE)

            # 前向传播
            pred_next_state_logits, pred_reward, pred_done_logits = model(states_onehot, actions_onehot)
            
            # 计算总损失
            loss_next_state = loss_fn_next_state(pred_next_state_logits, next_states)
            loss_reward = loss_fn_reward(pred_reward, rewards)
            loss_done = loss_fn_done(pred_done_logits, dones)
            total_loss = loss_next_state + loss_reward + loss_done
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())

        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {np.mean(epoch_losses):.4f}")

    print("世界模型训练完成！")

    # --- 5. 测试模型的“想象力” ---
    print("\n--- 测试模型预测能力 ---")
    state, info = env.reset()
    action = env.action_space.sample() # 随机选一个动作

    # 准备输入
    state_onehot = F.one_hot(torch.tensor([state]), num_classes=STATE_DIM).float().to(DEVICE)
    action_onehot = F.one_hot(torch.tensor([action]), num_classes=ACTION_DIM).float().to(DEVICE)

    # 使用模型进行“想象”
    model.eval() # 切换到评估模式
    with torch.no_grad():
        pred_ns_logits, pred_r, pred_d_logits = model(state_onehot, action_onehot)

    # 解读预测结果
    pred_next_state_prob = F.softmax(pred_ns_logits, dim=1).cpu().numpy().flatten()
    pred_reward = pred_r.cpu().item()
    pred_done_prob = F.sigmoid(pred_d_logits).cpu().item()

    # 找出最有可能的下一个状态
    predicted_next_state = np.argmax(pred_next_state_prob)

    print(f"当前状态: {state}, 采取动作: {action} (0:左, 1:下, 2:右, 3:上)")
    print(f"模型预测 ->")
    print(f"  最有可能的下一个状态: {predicted_next_state} (概率: {pred_next_state_prob[predicted_next_state]:.2f})")
    print(f"  预测的奖励: {pred_reward:.4f}")
    print(f"  预测的游戏结束概率: {pred_done_prob:.2f}")

    # <--- 在文件最末尾添加下面这行 --->
    torch.save(model.state_dict(), "frozenlake_world_model.pth")
    print("\n世界模型已保存到 frozenlake_world_model.pth")