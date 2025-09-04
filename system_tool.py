# ==================================
# 系统工具准备单元格 (在安装pettingzoo前运行一次)
# ==================================

# 1. (推荐) 更新软件包列表，确保我们能找到最新的工具
!apt-get update -y

# 2. 安装最核心的编译工具包 (包含gcc, g++, make等)
#    这相当于Linux版的“Visual C++ Build Tools”
!apt-get install -y build-essential

# 3. (可选但推荐) 我们顺便把之前在Windows上装过的SWIG也装上
#    以及另一个常用的编译工具cmake，以备不时之需
!apt-get install -y swig cmake

print("系统编译工具安装完成！现在可以安装PettingZoo了。")