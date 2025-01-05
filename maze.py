import random
from collections import deque


class Maze:
    def __init__(self, rows, cols, start=(0, 0), goal=None):
        """
        rows, cols: 迷宫的行列数
        start: 起点坐标
        goal:  终点坐标，不指定则默认为 (rows-1, cols-1)
        """
        self.rows = rows
        self.cols = cols
        self.start = start
        if goal is None:
            goal = (rows - 1, cols - 1)
        self.goal = goal

        # 初始地图全 0（全通），后续每个算法里会根据需要重新填充
        self.grid = [[0] * cols for _ in range(rows)]

    def is_valid(self, r, c):
        """判断(r,c)是否在迷宫范围内且不是墙（值为0）"""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def has_valid_path(self):
        """检查是否存在从起点到终点的有效路径"""
        if not self.is_valid(self.start[0], self.start[1]):
            return False
        if not self.is_valid(self.goal[0], self.goal[1]):
            return False

        visited = set()
        queue = deque([self.start])
        visited.add(self.start)

        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if self.is_valid(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def carve_space_around(self, cell, steps=1):
        """
        确保 cell 附近有一定数量的空间，避免把起点/终点埋在墙里。
        steps 可以视为想要 carve 的"半径"或"额外步数"。
        （示例中做法很简单：基于 BFS 随机拓展若干步以打通一些区域。）
        """
        (sr, sc) = cell
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            return

        visited = set()
        visited.add(cell)
        frontier = [cell]
        self.grid[sr][sc] = 0  # 起始格设为通

        # 根据地图大小和需求，建议 steps 不要过大
        for _ in range(steps):
            new_frontier = []
            for r, c in frontier:
                # 从该点再做随机拓展
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            self.grid[nr][nc] = 0
                            new_frontier.append((nr, nc))
            frontier = new_frontier

    def bridge_start_goal(self):
        """
        如果从 start 到 goal 不可达，则尝试在迷宫中"桥接"两片区域。
        改进版本：使用多点连接策略和智能路径规划。
        """

        # 首先 BFS 获取 start 区域
        def get_region_bfs(root):
            region = set()
            queue = deque([root])
            region.add(root)
            while queue:
                r, c = queue.popleft()
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if self.is_valid(nr, nc) and (nr, nc) not in region:
                        region.add((nr, nc))
                        queue.append((nr, nc))
            return region

        if not self.is_valid(*self.start):
            return
        if not self.is_valid(*self.goal):
            return

        start_region = get_region_bfs(self.start)
        if self.goal in start_region:
            return  # 已连通，不用桥接

        goal_region = get_region_bfs(self.goal)

        # 找到多个可能的连接点对
        connection_pairs = []
        for sr, sc in start_region:
            for gr, gc in goal_region:
                dist = abs(sr - gr) + abs(sc - gc)  # 曼哈顿距离
                connection_pairs.append(((sr, sc), (gr, gc), dist))

        # 按距离排序，选择前几个最近的点对
        connection_pairs.sort(key=lambda x: x[2])
        top_pairs = connection_pairs[: min(5, len(connection_pairs))]

        # 对每个点对尝试连接，直到成功
        for (sr, sc), (gr, gc), _ in top_pairs:
            if self.try_connect_points(sr, sc, gr, gc):
                break

    def try_connect_points(self, sr, sc, gr, gc):
        """使用改进的连接算法尝试连接两点"""

        # 使用A*启发式来引导路径
        def heuristic(r, c):
            return abs(r - gr) + abs(c - gc)

        visited = set()
        # 优先队列：(启发值, 当前坐标)
        queue = [(heuristic(sr, sc), (sr, sc))]
        came_from = {(sr, sc): None}

        while queue:
            _, (r, c) = min(queue)  # 取启发值最小的点
            queue.remove((_, (r, c)))

            if (r, c) == (gr, gc):
                # 找到路径，开始carve
                current = (r, c)
                while current in came_from and came_from[current] is not None:
                    self.grid[current[0]][current[1]] = 0
                    current = came_from[current]
                return True

            visited.add((r, c))

            # 尝试四个方向，但加入智能绕路
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)  # 增加一些随机性，避免路径过于规则

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and (nr, nc) not in visited
                ):
                    # 计算新的启发值
                    h_val = heuristic(nr, nc)

                    # 如果是墙，增加一些代价，但不要完全排除
                    if self.grid[nr][nc] == 1:
                        h_val += 2

                    queue.append((h_val, (nr, nc)))
                    came_from[(nr, nc)] = (r, c)

        return False

    def force_connect(self):
        """
        强制打通起点到终点的路径。
        原理：找到最近的两个点对，然后强制连接。
        """

        # 首先获取起点和终点所在的连通区域
        def get_region(start_point):
            region = set()
            queue = deque([start_point])
            region.add(start_point)

            while queue:
                r, c = queue.popleft()
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.rows
                        and 0 <= nc < self.cols
                        and self.grid[nr][nc] == 0
                        and (nr, nc) not in region
                    ):
                        region.add((nr, nc))
                        queue.append((nr, nc))
            return region

        # 获取两个区域
        start_region = get_region(self.start)
        if self.goal in start_region:  # 如果已经连通，直接返回
            return True

        goal_region = get_region(self.goal)

        # 找到两个区域之间的最近点对
        min_dist = float("inf")
        best_pair = None

        for sr, sc in start_region:
            for gr, gc in goal_region:
                dist = abs(sr - gr) + abs(sc - gc)  # 曼哈顿距离
                if dist < min_dist:
                    min_dist = dist
                    best_pair = ((sr, sc), (gr, gc))

        if best_pair is None:  # 这种情况理论上不会发生
            return False

        # 强制打通这两点之间的路径
        (sr, sc), (gr, gc) = best_pair

        # 使用直线路径打通
        r, c = sr, sc
        while r != gr or c != gc:
            # 优先在距离更大的方向移动
            if abs(r - gr) > abs(c - gc):
                r += 1 if gr > r else -1
            else:
                c += 1 if gc > c else -1
            self.grid[r][c] = 0  # 强制打通

            # 为了让路径看起来更自然，随机打通周围的一些格子
            if random.random() < 0.3:  # 30%的概率
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self.grid[nr][nc] = 0

        return True

    # ----------------------------------------------------------------
    # 下面是各类迷宫生成函数
    # ----------------------------------------------------------------

    def generate_traditional_maze(self, ensure_path=True):
        """
        使用 DFS 生成传统迷宫，但可以通过在 carve 过程中"随机跳过"来控制是否整体连通。
        ensure_path=True ->  最终会保证从 start 到 goal 连通（通过 bridge）。
        ensure_path=False -> 最终不一定连通。
        """
        # 初始化所有格子为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        def valid_cell(r, c):
            return 0 <= r < self.rows and 0 <= c < self.cols

        def carve_path(r, c, visited):
            visited.add((r, c))
            self.grid[r][c] = 0  # 打通当前格子

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dr, dc in directions:
                nr, nc = r + 2 * dr, c + 2 * dc
                if valid_cell(nr, nc) and (nr, nc) not in visited:
                    # 为了可能形成不连通区域，加入一个"随机放弃 carve"的机制
                    if random.random() < 0.9 or ensure_path:
                        # 打通中间墙
                        self.grid[r + dr][c + dc] = 0
                        carve_path(nr, nc, visited)

        # 将 start 作为 DFS 起点
        sr, sc = self.start
        if not valid_cell(sr, sc):
            sr, sc = 0, 0

        # 若起点在墙外，这里先修正一下
        sr = max(0, min(sr, self.rows - 1))
        sc = max(0, min(sc, self.cols - 1))
        carve_path(sr, sc, visited=set())

        # 若 ensure_path 为 True，则强制保证可达
        if ensure_path and not self.has_valid_path():
            self.force_connect()

        # 保证 start/goal 周边有一定空间
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)

    def generate_river_maze(self, ensure_path=True):
        """
        生成"河流"型迷宫，通道更狭窄且蜿蜒。
        若 ensure_path=True，则最终桥接 start->goal 。
        """
        # 先将全图置为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        def in_bounds(r, c):
            return 0 <= r < self.rows and 0 <= c < self.cols

        def carve_narrow_path(r, c, steps, visited):
            """
            从(r,c)开始随机步进 'steps' 步，形成窄且蜿蜒的路径。
            """
            stack = [(r, c)]
            visited.add((r, c))
            self.grid[r][c] = 0

            for _ in range(steps):
                if not stack:
                    break
                cr, cc = stack[-1]

                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                moved = False
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(nr, nc) and (nr, nc) not in visited:
                        # 限制通道宽度：检查周围空地数量
                        open_neighbors = 0
                        for rr in range(nr - 1, nr + 2):
                            for cc2 in range(nc - 1, nc + 2):
                                if in_bounds(rr, cc2) and self.grid[rr][cc2] == 0:
                                    open_neighbors += 1
                        if open_neighbors <= 2:
                            self.grid[nr][nc] = 0
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                            moved = True
                            break
                if not moved:
                    stack.pop()

        visited = set()
        # 在 (start) 附近开凿一条长河流，保证有机会从 start 出发
        sr, sc = self.start
        sr = max(0, min(sr, self.rows - 1))
        sc = max(0, min(sc, self.cols - 1))
        max_steps = self.rows * self.cols // 2
        carve_narrow_path(sr, sc, max_steps, visited)

        # 再随机生成一些小支流
        river_count = (self.rows * self.cols) // 80
        for _ in range(river_count):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            steps = random.randint(
                (self.rows + self.cols) // 4, (self.rows + self.cols) // 2
            )
            carve_narrow_path(r, c, steps, visited)

        # 强制保证可达
        if ensure_path and not self.has_valid_path():
            self.force_connect()

        # 保证 start/goal 周边有一定空间
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)

    def generate_random_maze(self, wall_ratio=0.3, ensure_path=True):
        """
        生成随机障碍物迷宫
        ensure_path = True -> 最后做桥接保证 start->goal 可达
        ensure_path = False -> 纯随机，不保证通路
        """
        self.grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                # 先保证 start/goal 自己是通的
                if (r, c) in [self.start, self.goal]:
                    row.append(0)
                else:
                    # 以 wall_ratio 的概率生成墙
                    row.append(1 if random.random() < wall_ratio else 0)
            self.grid.append(row)

        if ensure_path and not self.has_valid_path():
            self.force_connect()

        # 保证 start/goal 附近通
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)

    def generate_block_maze(self, min_size=3, max_size=6, ensure_path=True):
        """
        生成块状障碍物，类似于在空地上随机放置一些"巨石"或大方块。
        ensure_path=True -> 最后保证 start->goal 可达（桥接）
        ensure_path=False -> 不一定通
        """
        # 先把全图置为可通
        self.grid = [[0] * self.cols for _ in range(self.rows)]

        # 随机放置块状障碍
        block_count = (self.rows * self.cols) // 25
        for _ in range(block_count):
            size = random.randint(min_size, max_size)
            r = random.randint(0, self.rows - size)
            c = random.randint(0, self.cols - size)
            for i in range(size):
                for j in range(size):
                    rr = r + i
                    cc = c + j
                    # 如果正好是 start/goal，就不要覆盖成墙
                    if (rr, cc) not in (self.start, self.goal):
                        self.grid[rr][cc] = 1

        # 强制保证可达
        if ensure_path and not self.has_valid_path():
            self.force_connect()

        # 保证 start/goal 附近通
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)
