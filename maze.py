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

        # 默认生成传统迷宫，如果需要可自由切换到其他生成算法
        self.generate_traditional_maze(ensure_path=True)

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
        """如果迷宫中没有从start到goal的通路，则强行开辟一条直线路径"""
        if self.has_valid_path():
            return

        # 简单做法：强行打通起点到终点的直线（列优先或行优先均可）
        r, c = self.start
        end_r, end_c = self.goal
        while (r, c) != (end_r, end_c):
            self.grid[r][c] = 0
            if r < end_r:
                r += 1
            elif r > end_r:
                r -= 1
            elif c < end_c:
                c += 1
            elif c > end_c:
                c -= 1
        self.grid[end_r][end_c] = 0

    def generate_traditional_maze(self, ensure_path=True):
        """
        使用深度优先搜索(DFS)生成传统迷宫。
        ensure_path = True ->  迷宫整体连通，从start可达goal
        ensure_path = False -> 随机生成若干不连通的区域，可能无法从start到goal
        """
        # 初始化所有格子为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        if ensure_path:
            # --- 完整 DFS 确保整张迷宫连通 ---
            # 将start强制变为可通行
            sr, sc = self.start
            if 0 <= sr < self.rows and 0 <= sc < self.cols:
                self.grid[sr][sc] = 0

            def carve_path(r, c, visited):
                visited.add((r, c))
                self.grid[r][c] = 0  # 打通当前格子

                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                for dr, dc in directions:
                    nr, nc = r + dr * 2, c + dc * 2
                    if (
                        0 <= nr < self.rows
                        and 0 <= nc < self.cols
                        and (nr, nc) not in visited
                        and self.grid[nr][nc] == 1
                    ):
                        # 打通中间墙
                        self.grid[r + dr][c + dc] = 0
                        carve_path(nr, nc, visited)

            visited = set()
            carve_path(sr, sc, visited)

            # 如果还是不通，则再强制打通一条路径
            if not self.has_valid_path():
                self.ensure_path_exists()

        else:
            # --- 随机 DFS 分块生成若干不相连区域，并不保证 start -> goal 可达 ---
            def carve_isolated_region(sr, sc, max_cells):
                if self.grid[sr][sc] == 0:  # 如果起点已经是通路，则跳过
                    return

                count = 0
                stack = [(sr, sc)]
                visited_local = set()

                while stack and count < max_cells:
                    r, c = stack.pop()
                    if (r, c) in visited_local:
                        continue
                    visited_local.add((r, c))
                    self.grid[r][c] = 0
                    count += 1

                    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
                    random.shuffle(directions)
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        mid_r, mid_c = r + dr // 2, c + dc // 2
                        if (
                            0 <= nr < self.rows
                            and 0 <= nc < self.cols
                            and self.grid[nr][nc] == 1
                        ):
                            # 随机决定是否继续往该方向 carve
                            if random.random() < 0.7:
                                self.grid[mid_r][mid_c] = 0
                                stack.append((nr, nc))

            # 全墙初始化已经做过，这里做多个不相连区域
            region_count = random.randint(3, 6)
            for _ in range(region_count):
                # 随机找一个起点（偶数行、偶数列常见做法，可自行调整）
                sr = random.randrange(0, self.rows, 2)
                sc = random.randrange(0, self.cols, 2)
                max_region_size = random.randint(
                    (self.rows * self.cols) // 20,
                    (self.rows * self.cols) // 10,
                )
                carve_isolated_region(sr, sc, max_region_size)

            # 注意：此模式下不再保证 start 和 goal 一定是通的
            # 如果想让 start, goal 至少是空地，但不一定相互连通，可打开下面注释行
            # self.grid[self.start[0]][self.start[1]] = 0
            # self.grid[self.goal[0]][self.goal[1]] = 0

    def generate_river_maze(self, ensure_path=True):
        """
        生成“河流”型迷宫，使通道更狭窄且蜿蜒。
        ensure_path = True ->  从start蜿蜒走到goal，形成一条主河道
        ensure_path = False -> 只在地图随机处生成若干条蜿蜒水道，不保证与start或goal连通
        """
        # 先将全图置为墙
        self.grid = [[1] * self.cols for _ in range(self.rows)]

        def in_bounds(r, c):
            return 0 <= r < self.rows and 0 <= c < self.cols

        def carve_narrow_path(r, c, steps, visited, forced_end=False):
            """
            从(r,c)开始随机步进 'steps' 步，形成窄且蜿蜒的路径。
            forced_end=True 时，如果在steps内抵达goal则结束，否则继续随机直到步数结束。
            """
            stack = [(r, c)]
            visited.add((r, c))
            self.grid[r][c] = 0

            for _ in range(steps):
                if not stack:
                    break
                cr, cc = stack[-1]

                # 如果要求强制到goal并且当前到达goal，退出
                if forced_end and (cr, cc) == self.goal:
                    break

                # 当前方向倾向
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)

                moved = False
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    # 保证不越界、没有访问过、保持窄通道
                    if in_bounds(nr, nc) and (nr, nc) not in visited:
                        # 检查邻居，保证通道不要变宽
                        # 只允许通道本身以及当前点
                        open_neighbors = 0
                        for rr in range(nr - 1, nr + 2):
                            for cc2 in range(nc - 1, nc + 2):
                                if in_bounds(rr, cc2) and self.grid[rr][cc2] == 0:
                                    open_neighbors += 1
                        # 控制通道“宽度”
                        # 如果周围已经有较多空地，就跳过
                        if open_neighbors <= 2:
                            self.grid[nr][nc] = 0
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                            moved = True
                            break

                # 如果无法前进，就回溯
                if not moved:
                    stack.pop()

        if ensure_path:
            # 先在start->goal之间生成一条较长路径
            visited = set()
            # 理论上最多不会超过rows*cols步，但可根据需要调整
            max_steps = self.rows * self.cols
            carve_narrow_path(
                self.start[0], self.start[1], max_steps, visited, forced_end=True
            )
            # 如果还没通，就强行打通
            if not self.has_valid_path():
                self.ensure_path_exists()

        else:
            # 在地图中随机生成若干条河道
            river_count = (self.rows * self.cols) // 80  # 可根据尺寸灵活调节数量
            for _ in range(river_count):
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                visited = set()
                # 随机长度
                steps = random.randint(
                    (self.rows + self.cols) // 2, (self.rows + self.cols)
                )
                carve_narrow_path(r, c, steps, visited)

    def generate_random_maze(self, wall_ratio=0.3, ensure_path=True):
        """
        生成随机障碍物迷宫
        ensure_path = True -> 生成时确保存在一条有效路径
        ensure_path = False -> 纯随机，不保证有通路
        """
        if ensure_path:
            # 类似生长算法，先构建一个联通路径集，再随机决定哪些格子留空
            self.grid = [[1] * self.cols for _ in range(self.rows)]
            path_cells = set([self.start])
            frontier = set()

            # 添加起点周围的候选格子
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = self.start[0] + dr, self.start[1] + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    frontier.add((r, c))

            # 随机生长直到到达目标或无法继续
            while frontier and self.goal not in path_cells:
                current = random.choice(list(frontier))
                frontier.remove(current)

                # 检查是否有与现有路径相邻
                neighbors = []
                r, c = current
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in path_cells:
                        neighbors.append((nr, nc))

                # 有相邻路径时，随机决定是否留空
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

            # 如果目标仍不在通路中，则补救打通
            if self.goal not in path_cells:
                # 强制打通起点到终点
                self.ensure_path_exists()
        else:
            self.grid = []
            for r in range(self.rows):
                row = []
                for c in range(self.cols):
                    if (r, c) in (self.start, self.goal):
                        row.append(0)
                    else:
                        row.append(1 if random.random() < wall_ratio else 0)
                self.grid.append(row)

    def generate_block_maze(self, min_size=3, max_size=6, ensure_path=True):
        """
        生成块状障碍物，类似“巨石”分布。
        ensure_path=True -> 先确保起点到终点可达，再随机放置块状障碍
        ensure_path=False -> 纯随机放置块状障碍
        """
        self.grid = [[0] * self.cols for _ in range(self.rows)]

        if ensure_path:
            # 先打通一条基础路径
            current = self.start
            path = [current]
            while current != self.goal:
                cr, cc = current
                # 随机走向更接近goal的方向
                dr = 0
                dc = 0
                if cr < self.goal[0]:
                    dr = 1
                elif cr > self.goal[0]:
                    dr = -1
                elif cc < self.goal[1]:
                    dc = 1
                elif cc > self.goal[1]:
                    dc = -1
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    current = (nr, nc)
                    path.append(current)
                else:
                    break

            # 打通路径
            for r, c in path:
                self.grid[r][c] = 0

            # 在不破坏已通路的前提下放置障碍
            block_count = (self.rows * self.cols) // 25
            for _ in range(block_count):
                size = random.randint(min_size, max_size)
                placed = False
                for _attempt in range(50):
                    r = random.randint(0, self.rows - size)
                    c = random.randint(0, self.cols - size)
                    # 检查该区块是否会覆盖已有路径
                    conflict = False
                    for i in range(size):
                        for j in range(size):
                            if (r + i, c + j) in path:
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        # 放置障碍块
                        for i in range(size):
                            for j in range(size):
                                self.grid[r + i][c + j] = 1
                        placed = True
                        break
                # 如果50次都没放置成功就放弃该块
        else:
            # 纯随机放置块状障碍
            self.grid = [[0] * self.cols for _ in range(self.rows)]
            block_count = (self.rows * self.cols) // 25
            for _ in range(block_count):
                size = random.randint(min_size, max_size)
                r = random.randint(0, self.rows - size)
                c = random.randint(0, self.cols - size)
                for i in range(size):
                    for j in range(size):
                        if (r + i, c + j) not in (self.start, self.goal):
                            self.grid[r + i][c + j] = 1

    def is_valid(self, r, c):
        """判断(r,c)是否在迷宫范围内且不是墙（值为0）"""
        return (0 <= r < self.rows) and (0 <= c < self.cols) and (self.grid[r][c] == 0)


if __name__ == "__main__":
    # 简单测试
    maze = Maze(rows=21, cols=31)
    maze.generate_traditional_maze(ensure_path=False)  # 不保证 start->goal 有通路
    print("Maze with ensure_path=False (traditional):", maze.has_valid_path())

    maze.generate_traditional_maze(ensure_path=True)  # 保证可达到
    print("Maze with ensure_path=True (traditional):", maze.has_valid_path())

    maze.generate_river_maze(ensure_path=False)
    print("River maze (ensure_path=False):", maze.has_valid_path())

    maze.generate_river_maze(ensure_path=True)
    print("River maze (ensure_path=True):", maze.has_valid_path())
