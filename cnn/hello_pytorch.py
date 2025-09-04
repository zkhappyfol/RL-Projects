import torch
import numpy as np

print("===== 欢迎来到 PyTorch 的世界! =====\n")

# --- 1. 创建一个 Tensor ---
# 和创建 python 列表或 numpy 数组非常像
data = [[1, 2], [3, 4]]
x_tensor = torch.tensor(data)

print("这是一个 PyTorch Tensor:")
print(x_tensor)
print(f"Tensor 的数据类型: {x_tensor.dtype}")
print(f"Tensor 的形状: {x_tensor.shape}\n")


# --- 2. Tensor 与 NumPy 的无缝衔接 ---
# a. 从 NumPy 数组创建 Tensor
np_array = np.array(data)
x_from_numpy = torch.from_numpy(np_array)

print("这是一个从 NumPy 数组转换来的 Tensor:")
print(x_from_numpy)
print("--------------------")

# b. 从 Tensor 转换回 NumPy 数组
np_from_tensor = x_tensor.numpy()
print("这是一个从 Tensor 转换回的 NumPy 数组:")
print(np_from_tensor)
print("--------------------\n")


# --- 3. Tensor 的基本运算 ---
# 就像NumPy一样，可以进行各种数学运算
y_tensor = torch.tensor([[5, 6], [7, 8]])

print("两个 Tensor 相加:")
print(x_tensor + y_tensor)
print("\n一个 Tensor 乘以 2:")
print(x_tensor * 2)
print("\n")

# --- 4. 魔法的起点：梯度追踪 ---
# 创建一个Tensor并告诉PyTorch：“请追踪这个家伙身上发生的所有计算”
z_tensor = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)

print("这是一个需要追踪梯度的 Tensor:")
print(z_tensor)

# 经过一些计算后...
output = z_tensor.pow(2).sum() # 计算 z 中每个元素的平方，然后求和
print(f"对它进行一番计算后得到 output: {output}")
# PyTorch 已经默默记下了计算图，为“自动求导”做好了准备！
# 我们暂时不深入，先感受一下这个概念即可。

print("\n===== 探索结束! =====")