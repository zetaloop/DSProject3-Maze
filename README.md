# 迷宫寻路算法可视化演示（课程作业）

<img width="250" alt="Snipaste_2025-01-08_11-02-52" src="https://github.com/user-attachments/assets/3f793cb4-317a-4c25-9d7a-c52ebd8c87ea" />
<img width="250" alt="Snipaste_2025-01-08_11-01-02" src="https://github.com/user-attachments/assets/792abc6b-fe03-4338-82e4-4cec8a8d7772" />
<img width="250" alt="Snipaste_2025-01-08_11-04-30" src="https://github.com/user-attachments/assets/c81d3007-1c81-43cb-bb09-c09a9ab059b3" />

这是一个用Python开发的迷宫寻路算法可视化演示程序，旨在帮助用户直观地理解和比较不同寻路算法的工作原理和性能特点。

## 功能特性

### 迷宫生成
- 支持5种不同类型的迷宫地形：
  - 空白地形：无任何障碍，适合观察基本搜索模式
  - 随机障碍：随机分布的墙体，模拟自然环境
  - 传统迷宫：经典迷宫布局，由连续墙体构成
  - 块状地形：大块矩形障碍，模拟山地地形
  - 河流地形：蜿蜒曲折的狭窄通道

### 寻路算法
- 实现了5种经典寻路算法：
  - 深度优先搜索 (DFS)：优先探索深度方向，内存占用小
  - 广度优先搜索 (BFS)：逐层扩展搜索范围，保证最短路径
  - 双向广度优先搜索：同时从起点和终点搜索，效率更高
  - 贪心搜索：基于启发式估计选择路径，速度快
  - A*搜索：结合实际代价和启发式估计，寻找最优路径

### 可视化特性
- 实时动态展示搜索过程
- 使用不同颜色标识：
  - 已访问节点
  - 当前搜索前沿
  - 最终路径
- 支持调节演示速度
- 双倍地图尺寸模式
- 自适应深色/浅色主题

## 环境要求

- Python 3.12+
- 操作系统：Windows/macOS/Linux

## 快速开始

### 从源码运行

1. 克隆或下载本项目
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行程序：
   ```bash
   python main.py
   ```

### 使用可执行文件（Windows）

1. 运行 `make_bin.cmd` 生成可执行文件
2. 在 `dist` 目录下找到生成的exe文件并运行

## 使用说明

1. 选择迷宫类型：
   - 从下拉菜单选择想要的地形类型
   - 可选择是否确保起点到终点可达

2. 选择寻路算法：
   - 从下拉菜单选择要演示的算法
   - 每种算法都有详细说明

3. 调节演示：
   - 使用速度滑块控制演示速度
   - 可选择是否使用双倍地图尺寸

4. 操作按钮：
   - 生成新迷宫：重新生成当前类型的迷宫
   - 开始求解：启动寻路算法演示
   - 重置：清除当前搜索状态
   - 切换主题：在深色/浅色主题间切换

## 项目结构

- `main.py`：程序入口和GUI实现
- `maze.py`：迷宫生成算法
- `solver.py`：寻路算法实现
- `requirements.txt`：项目依赖
- `make_bin.cmd`：构建脚本

## 依赖说明

主要依赖包括：
- sv-ttk：现代化的ttk主题
- darkdetect：系统主题检测
- pyinstaller：用于构建可执行文件

## 注意事项

- 不同的迷宫类型适合展示不同算法的特点
- 双向BFS在大型迷宫中通常更高效
- A*算法在有明确目标时表现最好
- 可以通过调节速度来详细观察算法的工作过程
- 双倍尺寸模式可以更好地展示大规模寻路过程

## 构建说明

### Windows系统

运行 `make_bin.cmd` 脚本即可自动完成以下步骤：
1. 创建虚拟环境
2. 安装所需依赖
3. 使用PyInstaller打包
4. 清理临时文件

构建完成后，可执行文件将位于 `dist` 目录下。
