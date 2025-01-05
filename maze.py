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
        steps 可以视为想要 carve 的“半径”或“额外步数”。
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
        如果从 start 到 goal 不可达，则尝试在迷宫中“桥接”两片区域。
        做法大致：
        1. 找到 start 所在连通分量 S，goal 所在连通分量 G。
        2. 如果 S == G 则说明已连通，退出。
        3. 否则，从 S 和 G 中各拿一个边缘点，沿着两者间的直线或随机线路进行 carve，使得S与G合并。
        4. 或者做更复杂的多段式桥接，这里给出一个相对简单的示例。
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
        # 找两个最近点，分别在 start_region 和 goal_region 中
        # （这里可以更智能一些，比如先抽样再计算最短距离，这里简单实现“暴力找最小距离”的示例）
        min_dist = float("inf")
        pair_to_connect = (None, None)
        for sr, sc in start_region:
            for gr, gc in goal_region:
                dist = (sr - gr) ** 2 + (sc - gc) ** 2
                if dist < min_dist:
                    min_dist = dist
                    pair_to_connect = ((sr, sc), (gr, gc))

        # 如果找不到合适的点，说明没必要连通或其他情况
        if pair_to_connect == (None, None):
            return

        (sr, sc), (gr, gc) = pair_to_connect

        # 用一个简单的随机走近式 carve 来“桥接”两点
        # （替代原来的“一条笔直线”）
        r, c = sr, sc
        attempts = 0
        max_attempts = self.rows * self.cols  # 防止死循环
        while (r, c) != (gr, gc) and attempts < max_attempts:
            self.grid[r][c] = 0
            # 随机选择一个方向，让 (r,c) 向 (gr,gc) 移动
            dr = 0
            dc = 0
            if r < gr:
                dr = 1
            elif r > gr:
                dr = -1
            if c < gc:
                dc = 1
            elif c > gc:
                dc = -1

            # 有一定概率“拐个弯”，避免就是笔直
            if random.random() < 0.3:
                # 随机略微往其它方向偏移
                rand_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                dr2, dc2 = rand_dir
                # 50% 几率用这个拐弯
                if random.random() < 0.5:
                    dr, dc = dr2, dc2

            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                r, c = nr, nc
            attempts += 1

        # 最后再 carve 终点，以防止它是墙
        self.grid[gr][gc] = 0

    # ----------------------------------------------------------------
    # 下面是各类迷宫生成函数（示例）
    # ----------------------------------------------------------------

    def generate_traditional_maze(self, ensure_path=True):
        """
        使用 DFS 生成传统迷宫，但可以通过在 carve 过程中“随机跳过”来控制是否整体连通。
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
                    # 为了可能形成不连通区域，加入一个“随机放弃 carve”的机制
                    if random.random() < 0.5 or ensure_path:
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

        # 若 ensure_path 为 True，则桥接保证可达
        if ensure_path and not self.has_valid_path():
            self.bridge_start_goal()

        # 保证 start/goal 周边有一定空间
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)

    def generate_river_maze(self, ensure_path=True):
        """
        生成“河流”型迷宫，通道更狭窄且蜿蜒。
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

        # 桥接
        if ensure_path and not self.has_valid_path():
            self.bridge_start_goal()

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
            self.bridge_start_goal()

        # 保证 start/goal 附近通
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)

    def generate_block_maze(self, min_size=3, max_size=6, ensure_path=True):
        """
        生成块状障碍物，类似于在空地上随机放置一些“巨石”或大方块。
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

        # 如果要求可达，就桥接
        if ensure_path and not self.has_valid_path():
            self.bridge_start_goal()

        # 保证 start/goal 附近通
        self.carve_space_around(self.start, steps=1)
        self.carve_space_around(self.goal, steps=1)
