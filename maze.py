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

        self.generate_traditional_maze(ensure_path=True)  # 默认生成有路径的传统迷宫

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
        """生成随机障碍物迷宫
        如果ensure_path为True，则使用生长算法确保路径存在
        """
        if ensure_path:
            # 使用随机生长算法，确保路径存在
            self.grid = [[1] * self.cols for _ in range(self.rows)]
            path_cells = set([self.start])
            frontier = set()

            # 添加起点周围的候选格子
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = self.start[0] + dr, self.start[1] + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    frontier.add((r, c))

            # 随机生长直到达到目标或无法继续
            while frontier and self.goal not in path_cells:
                # 随机选择一个候选格子
                current = random.choice(list(frontier))
                frontier.remove(current)

                # 检查是否有相邻的路径格子
                neighbors = []
                r, c = current
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in path_cells:
                        neighbors.append((nr, nc))

                # 如果有相邻路径，有一定概率将当前格子变为路径
                if neighbors and random.random() > wall_ratio:
                    path_cells.add(current)
                    self.grid[r][c] = 0

                    # 添加新的候选格子
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self.rows
                            and 0 <= nc < self.cols
                            and (nr, nc) not in path_cells
                            and (nr, nc) not in frontier
                        ):
                            frontier.add((nr, nc))

            # 确保终点可达
            if self.goal not in path_cells:
                # 从终点反向生长直到连接到现有路径
                current = self.goal
                path_cells.add(current)
                self.grid[current[0]][current[1]] = 0
                while not any(
                    (nr, nc) in path_cells
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    for nr, nc in [(current[0] + dr, current[1] + dc)]
                    if 0 <= nr < self.rows and 0 <= nc < self.cols
                ):
                    # 选择向现有路径最近的方向
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    min_dist = float("inf")
                    next_pos = None

                    for dr, dc in directions:
                        r, c = current[0] + dr, current[1] + dc
                        if 0 <= r < self.rows and 0 <= c < self.cols:
                            # 计算到最近路径点的距离
                            dist = min(
                                abs(r - pr) + abs(c - pc) for pr, pc in path_cells
                            )
                            if dist < min_dist:
                                min_dist = dist
                                next_pos = (r, c)

                    if next_pos:
                        current = next_pos
                        path_cells.add(current)
                        self.grid[current[0]][current[1]] = 0
        else:
            # 完全随机生成，不保证路径存在
            self.grid = [[0] * self.cols for _ in range(self.rows)]
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) not in (self.start, self.goal):
                        self.grid[r][c] = 1 if random.random() < wall_ratio else 0

    def generate_traditional_maze(self, ensure_path=True):
        """生成传统迷宫（使用深度优先搜索的随机迷宫生成算法）

        Args:
            ensure_path: 是否确保存在从起点到终点的路径
                        True - 使用标准的DFS算法，保证迷宫完全连通
                        False - 可能生成不连通的迷宫
        """
        # 初始化所有格子为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        if ensure_path:
            # 使用标准DFS生成有路径的迷宫
            def carve_path(r, c, visited):
                visited.add((r, c))
                self.grid[r][c] = 0  # 打通当前格子

                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                for dr, dc in directions:
                    new_r, new_c = r + dr * 2, c + dc * 2
                    if (
                        0 <= new_r < self.rows
                        and 0 <= new_c < self.cols
                        and (new_r, new_c) not in visited
                        and self.grid[new_r][new_c] == 1
                    ):
                        self.grid[r + dr][c + dc] = 0  # 打通中间的墙
                        carve_path(new_r, new_c, visited)

            visited = set()
            carve_path(self.start[0], self.start[1], visited)

        else:
            # 生成多个不相连的区域
            def carve_isolated_region(start_r, start_c, max_cells):
                if self.grid[start_r][start_c] == 0:  # 如果起点已经是通路，返回
                    return

                cells_carved = 0
                stack = [(start_r, start_c)]
                visited = set()

                while stack and cells_carved < max_cells:
                    r, c = stack.pop()
                    if (r, c) in visited:
                        continue

                    visited.add((r, c))
                    self.grid[r][c] = 0
                    cells_carved += 1

                    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
                    random.shuffle(directions)

                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        mid_r, mid_c = r + dr // 2, c + dc // 2

                        if (
                            0 <= new_r < self.rows
                            and 0 <= new_c < self.cols
                            and self.grid[new_r][new_c] == 1
                            and random.random() < 0.7  # 70%概率继续扩展
                        ):
                            self.grid[mid_r][mid_c] = 0
                            stack.append((new_r, new_c))

            # 确保起点和终点是通路
            self.grid[self.start[0]][self.start[1]] = 0
            self.grid[self.goal[0]][self.goal[1]] = 0

            # 随机选择多个起点生成不相连的区域
            num_regions = random.randint(3, 6)  # 生成3-6个不相连的区域
            for _ in range(num_regions):
                start_r = random.randrange(0, self.rows, 2)
                start_c = random.randrange(0, self.cols, 2)
                max_region_size = random.randint(
                    (self.rows * self.cols) // 20,  # 最小区域大小
                    (self.rows * self.cols) // 10,  # 最大区域大小
                )
                carve_isolated_region(start_r, start_c, max_region_size)

    def generate_block_maze(self, min_size=3, max_size=6, ensure_path=True):
        """生成大块地形的迷宫
        如果ensure_path为True，则先生成路径再添加障碍块
        """
        self.grid = [[0] * self.cols for _ in range(self.rows)]

        if ensure_path:
            # 先生成一条随机路径
            current = self.start
            path = set([current])
            while current != self.goal:
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)
                moved = False
                for dr, dc in directions:
                    new_r, new_c = current[0] + dr, current[1] + dc
                    if (
                        0 <= new_r < self.rows
                        and 0 <= new_c < self.cols
                        and (new_r, new_c) not in path
                        and abs(new_r - self.goal[0]) + abs(new_c - self.goal[1])
                        < abs(current[0] - self.goal[0])
                        + abs(current[1] - self.goal[1])
                    ):
                        current = (new_r, new_c)
                        path.add(current)
                        moved = True
                        break

                if not moved:  # 如果无法前进，回退一步
                    path.remove(current)
                    if not path:  # 如果无法生成路径，跳出
                        break
                    current = random.choice(list(path))

            # 在不影响路径的情况下添加障碍块
            for _ in range((self.rows * self.cols) // 25):
                size = random.randint(min_size, max_size)
                attempts = 50  # 最大尝试次数
                while attempts > 0:
                    r = random.randint(0, self.rows - size)
                    c = random.randint(0, self.cols - size)

                    # 检查是否会影响现有路径
                    can_place = True
                    for i in range(size):
                        for j in range(size):
                            if (r + i, c + j) in path:
                                can_place = False
                                break
                        if not can_place:
                            break

                    if can_place:
                        for i in range(size):
                            for j in range(size):
                                if 0 <= r + i < self.rows and 0 <= c + j < self.cols:
                                    self.grid[r + i][c + j] = 1
                        break

                    attempts -= 1
        else:
            # 直接放置随机大小的障碍块
            for _ in range((self.rows * self.cols) // 25):
                size = random.randint(min_size, max_size)
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                for i in range(size):
                    for j in range(size):
                        if (
                            0 <= r + i < self.rows
                            and 0 <= c + j < self.cols
                            and (r + i, c + j) not in (self.start, self.goal)
                        ):
                            self.grid[r + i][c + j] = 1

    def generate_river_maze(self, ensure_path=True):
        """生成河流型迷宫，包含蜿蜒的通道
        通过先填充墙壁，然后生成蜿蜒的通路来实现河流效果
        """
        # 初始化为全墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        def create_river_path(r, c, remaining_length, visited):
            if remaining_length <= 0 or not (0 <= r < self.rows and 0 <= c < self.cols):
                return False

            if (r, c) == self.goal:  # 如果到达终点，标记找到路径
                self.grid[r][c] = 0
                return True

            if (r, c) in visited:  # 避免路径交叉
                return False

            visited.add((r, c))
            self.grid[r][c] = 0  # 开辟通路

            # 偏向继续当前方向，使路径更平滑
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if len(visited) > 1:
                last_r, last_c = list(visited)[-2]
                current_dir = (r - last_r, c - last_c)
                # 调整方向权重，使其倾向于保持当前方向
                weights = [3 if d == current_dir else 1 for d in directions]
            else:
                weights = [1] * 4

            # 随机尝试方向
            directions = random.choices(directions, weights=weights, k=len(directions))
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                # 检查新位置的8个邻居，确保路径不会太宽
                is_path_valid = True
                for check_r in range(new_r - 1, new_r + 2):
                    for check_c in range(new_c - 1, new_c + 2):
                        if (
                            (check_r, check_c) != (r, c)
                            and 0 <= check_r < self.rows
                            and 0 <= check_c < self.cols
                        ):
                            if (
                                self.grid[check_r][check_c] == 0
                                and (check_r, check_c) not in visited
                            ):
                                is_path_valid = False
                                break
                    if not is_path_valid:
                        break

                if is_path_valid and create_river_path(
                    new_r, new_c, remaining_length - 1, visited
                ):
                    return True

            return False

        # 确保起点可用
        self.grid[self.start[0]][self.start[1]] = 0

        if ensure_path:
            # 尝试生成一条到达终点的主河道
            visited = set([self.start])
            max_length = self.rows * self.cols  # 最大长度限制
            if not create_river_path(self.start[0], self.start[1], max_length, visited):
                # 如果无法生成到终点的路径，回退到简单路径
                self.ensure_path_exists()
        else:
            # 生成多条不一定连通的河道
            num_rivers = self.rows // 3
            for _ in range(num_rivers):
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                visited = set()
                create_river_path(r, c, self.rows // 2, visited)

    def is_valid(self, r, c):
        """判断是否在迷宫范围内且不是墙"""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 0
        return False
