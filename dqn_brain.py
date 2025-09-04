import torch
import torch.nn as nn

# ==================================
# Part 1: 定义神经网络的“图纸”
# ==================================

# 我们创建一个类，来表示我们的“大脑”结构
# 它必须继承自 torch.nn.Module
class DQN(nn.Module):
    
    def __init__(self, input_size, output_size):
        # 调用父类的构造函数
        super(DQN, self).__init__()
        
        # 定义神经网络的结构，就像搭乐高一样
        # self.net 就是我们的“流水线”
        self.net = nn.Sequential(
            # 第1层：线性层。输入维度是环境状态大小，输出维度是128（可以自定义）
            nn.Linear(input_size, 128),
            # 激活函数：给网络增加非线性，让它能学习更复杂的关系
            nn.ReLU(),
            
            # 第2层：线性层。输入维度是上一层的输出128，输出维度是64
            nn.Linear(128, 64),
            nn.ReLU(),

            # 输出层：输入维度是上一层的输出64，输出维度是动作空间大小
            nn.Linear(64, output_size)
        )

    # 定义“信息”是如何在这条流水线上传播的（前向传播）
    def forward(self, x):
        # 输入 x (状态) 流过我们定义的 self.net 流水线
        return self.net(x)


# ==================================
# Part 2: “实例化”并测试我们的大脑
# ==================================

# 1. 定义我们环境的参数
# CartPole 的状态空间大小是 4
state_size = 4
# CartPole 的动作空间大小是 2
action_size = 2

# 2. 根据图纸，创建我们的大脑实例
# 这就像根据乐高图纸，拼出一个真实的模型
brain = DQN(state_size, action_size)
print("我们创建的大脑结构如下：")
print(brain)
print("\n")

# 3. 模拟一次“思考”过程
# 假设我们从环境中得到了一个状态
# 注意：状态需要是一个 Tensor
mock_state = torch.tensor([-0.03, 0.01, 0.04, -0.02], dtype=torch.float32)

# 把这个状态输入到大脑中，看看它会输出什么
# .no_grad() 表示这里我们只是在“测试”，不是在“学习”，不需要计算梯度
with torch.no_grad():
    q_values = brain(mock_state)

print(f"输入一个模拟的状态: {mock_state}")
print(f"大脑输出的Q值 (估算): {q_values}")
print(f"输出的Q值形状: {q_values.shape}")
print("\n")
print("可以看到，它为我们的2个动作（向左/向右）分别估算了一个Q值。")
print("现在这些值是随机的，因为大脑还没有经过任何训练！")