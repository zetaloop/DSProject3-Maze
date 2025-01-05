import random
from collections import deque


class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0] * cols for _ in range(rows)]

        # 起点和目标位置可根据需求随机生成或指定
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)

        self.generate_traditional_maze()  # 默认生成传统迷宫

    def has_valid_path(self):
        """检查是否存在从起点到终点的有效路径"""
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)

        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True

            # 检查四个方向
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_r, new_c = r + dr, c + dc
                if (
                    0 <= new_r < self.rows
                    and 0 <= new_c < self.cols
                    and self.grid[new_r][new_c] == 0
                    and (new_r, new_c) not in visited
                ):
                    queue.append((new_r, new_c))
                    visited.add((new_r, new_c))

        return False

    def ensure_path_exists(self):
        """确保存在一条从起点到终点的路径"""
        if self.has_valid_path():
            return

        # 如果没有路径，使用BFS创建一条
        visited = set()
        queue = deque([(self.start, [self.start])])
        visited.add(self.start)

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.goal:
                # 找到路径，清除这条路上的所有障碍
                for pr, pc in path:
                    self.grid[pr][pc] = 0
                return

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_r, new_c = r + dr, c + dc
                if (
                    0 <= new_r < self.rows
                    and 0 <= new_c < self.cols
                    and (new_r, new_c) not in visited
                ):
                    queue.append(((new_r, new_c), path + [(new_r, new_c)]))
                    visited.add((new_r, new_c))

        # 如果还是找不到路径（理论上不会发生），清除起点到终点的直线路径
        r, c = self.start
        end_r, end_c = self.goal
        while (r, c) != (end_r, end_c):
            self.grid[r][c] = 0
            if r < end_r:
                r += 1
            elif r > end_r:
                r -= 1
            if c < end_c:
                c += 1
            elif c > end_c:
                c -= 1
        self.grid[end_r][end_c] = 0

    def generate_random_maze(self, wall_ratio=0.3, ensure_path=True):
        """生成随机障碍物迷宫"""
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.start or (r, c) == self.goal:
                    self.grid[r][c] = 0
                else:
                    self.grid[r][c] = 1 if random.random() < wall_ratio else 0

        if ensure_path:
            self.ensure_path_exists()

    def generate_traditional_maze(self):
        """生成传统迷宫（使用深度优先搜索的随机迷宫生成算法）"""
        # 传统迷宫总是有路径的，不需要确保路径存在
        # 初始化所有格子为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        # 确保起点可用
        self.grid[self.start[0]][self.start[1]] = 0

        def carve_path(r, c):
            # 定义四个方向：上、右、下、左
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dr, dc in directions:
                new_r, new_c = r + dr * 2, c + dc * 2
                # 修改边界检查逻辑，允许到达最后一行和最后一列
                if (
                    0 <= new_r < self.rows
                    and 0 <= new_c < self.cols
                    and self.grid[new_r][new_c] == 1
                ):
                    # 打通中间的墙
                    self.grid[r + dr][c + dc] = 0
                    self.grid[new_r][new_c] = 0  # 打通目标格子
                    carve_path(new_r, new_c)

        # 从起点开始生成迷宫
        carve_path(self.start[0], self.start[1])

        # 确保终点可达
        # 如果终点附近都是墙，打通一条路径
        goal_r, goal_c = self.goal
        if self.grid[goal_r][goal_c] == 1:
            # 尝试从终点向四个方向找最近的通路
            for dr, dc in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
                r, c = goal_r + dr, goal_c + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.grid[r][c] = 0
                    if self.has_valid_path():  # 如果打通后有路径，就完成
                        break
            # 最后设置终点为通路
            self.grid[goal_r][goal_c] = 0

    def generate_block_maze(self, min_size=3, max_size=6, ensure_path=True):
        """生成大块地形的迷宫"""
        self.grid = [[0] * self.cols for _ in range(self.rows)]

        def place_block(r, c, size):
            for i in range(size):
                for j in range(size):
                    if (
                        0 <= r + i < self.rows
                        and 0 <= c + j < self.cols
                        and (r + i, c + j) != self.start
                        and (r + i, c + j) != self.goal
                    ):
                        self.grid[r + i][c + j] = 1

        # 随机放置大块障碍物
        num_blocks = (self.rows * self.cols) // 25  # 控制大块数量
        for _ in range(num_blocks):
            size = random.randint(min_size, max_size)
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            place_block(r, c, size)

        if ensure_path:
            self.ensure_path_exists()

    def generate_river_maze(self, ensure_path=True):
        """生成河流型迷宫，包含蜿蜒的通道"""
        self.grid = [[0] * self.cols for _ in range(self.rows)]

        def create_river(r, c, length):
            if length <= 0:
                return
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return

            self.grid[r][c] = 1
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            weights = [0.3, 0.3, 0.2, 0.2]  # 偏向向右和向下流动
            dr, dc = random.choices(directions, weights=weights)[0]
            create_river(r + dr, c + dc, length - 1)

        # 创建多条河流
        num_rivers = self.rows // 3
        for _ in range(num_rivers):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            length = random.randint(self.rows // 2, self.rows)
            create_river(r, c, length)

        # 确保起点和终点可达
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0

        if ensure_path:
            self.ensure_path_exists()

    def is_valid(self, r, c):
        """判断是否在迷宫范围内且不是墙"""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 0
        return False
