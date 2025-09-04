import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 我们的CNN大脑结构 ---
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        # input_shape 会是 (4, 84, 84) -> (通道数, 高, 宽)
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # 卷积层：像“侦探”一样提取图像特征
        self.conv_layers = nn.Sequential(
            # 第一个卷积层：输入4个通道(堆叠的帧)
            # 输出32个特征图，卷积核大小8x8，步长4
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 第二个卷积层：输入32个通道
            # 输出64个特征图，卷积核大小4x4，步长2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 第三个卷积层：输入64个通道
            # 输出64个特征图，卷积核大小3x3，步长1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 为了知道卷积层输出的大小，我们需要做一个“虚拟前向传播”
        # 这样可以动态计算出第一个全连接层的输入大小
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)
        
        # 全连接层：像“指挥官”一样根据特征做决策
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, x):
        """计算卷积层输出的扁平化尺寸"""
        o = self.conv_layers(x)
        return int(torch.flatten(o, 1).size(1))

    def forward(self, x):
        """定义数据如何流经网络"""
        # PyTorch的卷积层需要输入形状为 (N, C, H, W)
        # N: 批量大小, C: 通道数, H: 高, W: 宽
        # 我们的输入是 (N, H, W, C)，所以需要调整维度顺序
        # x = x.permute(0, 3, 1, 2) # 如果输入是 (N,H,W,C)
        
        conv_out = self.conv_layers(x)
        
        # 将卷积层的输出“压平”成一维向量
        flattened = torch.flatten(conv_out, 1)
        
        # 输入到全连接层，得到最终的Q值
        q_values = self.fc_layers(flattened)
        return q_values

# --- 测试我们的CNN大脑 ---
if __name__ == '__main__':
    # 我们的状态形状是 (4, 84, 84)
    # PyTorch的习惯是 Channels-First (通道在前)
    INPUT_SHAPE = (4, 84, 84)
    # Pong有6个动作
    NUM_ACTIONS = 6
    
    # 创建大脑实例
    brain = CnnDQN(INPUT_SHAPE, NUM_ACTIONS)
    print("CNN大脑结构:")
    print(brain)
    
    # 创建一个假的“状态”输入 (批量大小为1)
    # 形状为 (1, 4, 84, 84)
    dummy_state = torch.rand(1, *INPUT_SHAPE)
    
    # 测试前向传播
    with torch.no_grad():
        q_values = brain(dummy_state)
        
    print(f"\n输入一个形状为 {dummy_state.shape} 的虚拟状态")
    print(f"大脑输出的Q值形状: {q_values.shape}")
    print(f"期望的输出形状: torch.Size([1, {NUM_ACTIONS}])")
    
    # 检查输出形状是否正确
    assert q_values.shape == (1, NUM_ACTIONS)
    print("\n形状检查通过！大脑基本结构设计正确。")