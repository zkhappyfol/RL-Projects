import torch

# 只需要加载文件，就能看到它的内部结构
# 它本质上是一个Python字典
model_weights = torch.load("ddpg_pendulum_actor.pth")

# 打印出所有参数层的名字 (字典的键)
print(model_weights.keys())