import heapq
from collections import deque


class BaseSolver:
    """所有寻路算法的基类。"""

    def __init__(self, maze):
        self.maze = maze
        self.start = maze.start
        self.goal = maze.goal
        self.visited = set()
        self.frontier_set = set()  # 用于可视化或调试
        self.came_from = {}
        self.path = []
        self.current_position = None  # 添加当前位置属性

    def reset(self):
        """重置算法状态"""
        self.visited = set()
        self.frontier_set = set()
        self.came_from = {}
        self.path = []
        self.current_position = None  # 重置当前位置

    def build_path(self, current):
        """根据 came_from 中的记录自终点反向构建路径"""
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        self.path = path


# ---------------------------------------------------------------------------
# 1. DFS 深度优先搜索
# ---------------------------------------------------------------------------
class DFSSolver(BaseSolver):
    """深度优先搜索：不保证最短路径，可能比较快找到一条路"""

    def __init__(self, maze):
        super().__init__(maze)
        self.stack = []
        self.reset()

    def reset(self):
        super().reset()
        self.stack = [self.start]
        self.frontier_set.add(self.start)

    def step(self):
        if not self.stack:
            return True  # 栈空，搜索终止，可能找不到目标

        current = self.stack.pop()
        self.current_position = current  # 更新当前位置
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True  # 找到目标

        self.visited.add(current)

        # 朝四个方向探索
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if (
                self.maze.is_valid(nr, nc)
                and (nr, nc) not in self.visited
                and (nr, nc) not in self.frontier_set
            ):
                self.stack.append((nr, nc))
                self.frontier_set.add((nr, nc))
                self.came_from[(nr, nc)] = current

        return False


# ---------------------------------------------------------------------------
# 2. BFS 广度优先搜索
# ---------------------------------------------------------------------------
class BFSSolver(BaseSolver):
    """广度优先搜索：在无权迷宫中可以保证找到最短路径"""

    def __init__(self, maze):
        super().__init__(maze)
        self.queue = deque()
        self.reset()

    def reset(self):
        super().reset()
        self.queue = deque([self.start])
        self.frontier_set.add(self.start)

    def step(self):
        if not self.queue:
            return True  # 队列空，说明搜索完毕或无解

        current = self.queue.popleft()
        self.current_position = current  # 更新当前位置
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True  # 找到目标

        self.visited.add(current)

        # 扩展四个方向
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if (
                self.maze.is_valid(nr, nc)
                and (nr, nc) not in self.visited
                and (nr, nc) not in self.frontier_set
            ):
                self.queue.append((nr, nc))
                self.frontier_set.add((nr, nc))
                self.came_from[(nr, nc)] = current

        return False


# ---------------------------------------------------------------------------
# 3. Bidirectional BFS 双向广度优先搜索
# ---------------------------------------------------------------------------
class BidirectionalBFSSolver(BaseSolver):
    """
    双向广度优先搜索：从起点和终点分别向中间搜索，
    当两个方向相遇时即可拼接出完整路径。对于大的稀疏迷宫通常更高效。
    """

    def __init__(self, maze):
        super().__init__(maze)
        self.start_queue = deque()
        self.goal_queue = deque()
        self.start_visited = set()
        self.goal_visited = set()
        self.start_came_from = {}
        self.goal_came_from = {}
        self.meeting_point = None
        self.is_expanding_start = True  # 添加标记来指示当前是否在扩展起点
        self.reset()

    def reset(self):
        super().reset()
        self.start_queue = deque([self.start])
        self.goal_queue = deque([self.goal])
        self.start_visited = {self.start}
        self.goal_visited = {self.goal}
        self.start_came_from = {}
        self.goal_came_from = {}
        self.meeting_point = None
        self.is_expanding_start = True  # 重置为从起点开始
        # 用于可视化的 frontier_set，可以先把起点、终点都放进去
        self.frontier_set = {self.start, self.goal}

    def build_bidirectional_path(self):
        """拼接双向搜索的路径"""
        # 从 meeting_point 往回走到 start
        path_from_start = []
        cur = self.meeting_point
        while cur in self.start_came_from:
            path_from_start.append(cur)
            cur = self.start_came_from[cur]
        path_from_start.append(self.start)
        path_from_start.reverse()

        # 从 meeting_point 往前走到 goal
        path_to_goal = []
        cur = self.goal_came_from[self.meeting_point]  # 从 meeting_point 的下一个点开始
        while cur != self.goal:  # 直到到达终点
            path_to_goal.append(cur)
            cur = self.goal_came_from[cur]
        path_to_goal.append(self.goal)  # 添加终点

        # 合并路径
        self.path = path_from_start + path_to_goal

    def expand_front(self, queue, visited, other_visited, came_from):
        """
        扩展一端队列的通用逻辑。
        如果遇到对方已访问过的节点，则返回该节点(相遇)；否则返回 None。
        """
        if not queue:
            return None

        current = queue.popleft()
        self.current_position = current  # 更新当前位置
        self.frontier_set.discard(current)
        self.visited.add(current)  # 添加到总的已访问集合中

        if current in other_visited:
            return current  # 相遇

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = current
                queue.append((nr, nc))
                self.frontier_set.add((nr, nc))

        return None

    def step(self):
        if not self.start_queue and not self.goal_queue:
            return True  # 均空，搜索结束

        # 交替扩展两端
        if self.is_expanding_start:
            if self.start_queue:  # 如果起点队列非空，则扩展起点
                meeting = self.expand_front(
                    self.start_queue,
                    self.start_visited,
                    self.goal_visited,
                    self.start_came_from,
                )
                if meeting is not None:
                    self.meeting_point = meeting
                    self.build_bidirectional_path()
                    return True
            self.is_expanding_start = False  # 切换到终点
        else:
            if self.goal_queue:  # 如果终点队列非空，则扩展终点
                meeting = self.expand_front(
                    self.goal_queue,
                    self.goal_visited,
                    self.start_visited,
                    self.goal_came_from,
                )
                if meeting is not None:
                    self.meeting_point = meeting
                    self.build_bidirectional_path()
                    return True
            self.is_expanding_start = True  # 切换到起点

        return False


# ---------------------------------------------------------------------------
# 4. Greedy (Best-First) 搜索
# ---------------------------------------------------------------------------
class GreedySolver(BaseSolver):
    """
    贪心搜索：只看启发式 h(n) 最小的下一个节点，不累加代价。
    因此不能保证路径最短，但在稀疏迷宫中可能很快接近目标。
    """

    def __init__(self, maze):
        super().__init__(maze)
        self.frontier = []
        self.reset()

    def reset(self):
        super().reset()
        h_start = self.heuristic(self.start, self.goal)
        self.frontier = [(h_start, self.start)]
        self.frontier_set.add(self.start)

    def heuristic(self, pos1, pos2):
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        if not self.frontier:
            return True

        _, current = heapq.heappop(self.frontier)
        self.current_position = current  # 更新当前位置
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True

        self.visited.add(current)

        # 扩展相邻节点
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in self.visited:
                self.visited.add((nr, nc))  # 将新节点加入 visited
                h_val = self.heuristic((nr, nc), self.goal)
                heapq.heappush(self.frontier, (h_val, (nr, nc)))
                self.frontier_set.add((nr, nc))
                self.came_from[(nr, nc)] = current

        return False


# ---------------------------------------------------------------------------
# 5. A* 寻路
# ---------------------------------------------------------------------------
class AStarSolver(BaseSolver):
    """
    A* 搜索 - 对无权迷宫来说，g(n)+h(n) 也能找到最短路径；
    常用的启发式函数是曼哈顿距离。
    """

    def __init__(self, maze):
        super().__init__(maze)
        self.dist = {}  # g(n)：起点到当前节点的真实代价
        self.frontier = []  # 存 (f(n), g(n), (r, c))
        self.reset()

    def reset(self):
        super().reset()
        self.dist = {
            (r, c): float("inf")
            for r in range(self.maze.rows)
            for c in range(self.maze.cols)
        }
        self.dist[self.start] = 0
        # 初始节点的 f = g(=0) + h
        f_start = self.heuristic(self.start, self.goal)
        self.frontier = [(f_start, 0, self.start)]
        self.frontier_set.add(self.start)

    def heuristic(self, pos1, pos2):
        """使用曼哈顿距离作为启发式函数"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        if not self.frontier:
            return True  # 无路可走

        f, g, current = heapq.heappop(self.frontier)
        self.current_position = current  # 更新当前位置
        self.frontier_set.discard(current)

        # 如果当前节点就是目标，则构建路径
        if current == self.goal:
            self.build_path(current)
            return True

        # 若该节点对应的最优 g 值已经被更新过，则当前弹出的不再是最优
        if g > self.dist[current]:
            return False

        self.visited.add(current)

        # 扩展四个方向
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc):
                new_g = g + 1  # 无权，移动代价=1
                if new_g < self.dist[(nr, nc)]:
                    self.dist[(nr, nc)] = new_g
                    self.came_from[(nr, nc)] = current
                    f_new = new_g + self.heuristic((nr, nc), self.goal)
                    heapq.heappush(self.frontier, (f_new, new_g, (nr, nc)))
                    self.frontier_set.add((nr, nc))

        return False


# ---------------------------------------------------------------------------
# 6. Bidirectional A*
# ---------------------------------------------------------------------------
class BidirectionalAStarSolver(BaseSolver):
    """
    双向 A* - 从起点和终点同时搜索，
    特别适合起点和终点距离较远的迷宫环境。
    """

    def __init__(self, maze):
        super().__init__(maze)
        self.start_frontier = []
        self.goal_frontier = []
        self.start_dist = {}
        self.goal_dist = {}
        self.start_came_from = {}  # 分别存储两个方向的 came_from
        self.goal_came_from = {}
        self.meeting_point = None
        self.reset()

    def reset(self):
        super().reset()
        self.start_dist = {
            (r, c): float("inf")
            for r in range(self.maze.rows)
            for c in range(self.maze.cols)
        }
        self.goal_dist = {
            (r, c): float("inf")
            for r in range(self.maze.rows)
            for c in range(self.maze.cols)
        }
        self.start_dist[self.start] = 0
        self.goal_dist[self.goal] = 0
        self.start_frontier = [(self.heuristic(self.start, self.goal), 0, self.start)]
        self.goal_frontier = [(self.heuristic(self.goal, self.start), 0, self.goal)]
        self.start_came_from = {}
        self.goal_came_from = {}
        self.frontier_set = {self.start, self.goal}
        self.meeting_point = None

    def heuristic(self, pos1, pos2):
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def build_bidirectional_path(self):
        """拼接双向搜索的路径"""
        # 从 meeting_point 往回走到 start
        path_from_start = []
        cur = self.meeting_point
        while cur in self.start_came_from:
            path_from_start.append(cur)
            cur = self.start_came_from[cur]
        path_from_start.append(self.start)
        path_from_start.reverse()

        # 从 meeting_point 往前走到 goal
        path_to_goal = []
        cur = self.meeting_point
        while cur in self.goal_came_from:
            cur = self.goal_came_from[cur]
            path_to_goal.append(cur)

        # 合并路径，注意不要重复添加 meeting_point
        self.path = path_from_start + path_to_goal

    def step(self):
        if not self.start_frontier and not self.goal_frontier:
            return True  # 无解

        if len(self.start_frontier) <= len(self.goal_frontier):
            # 扩展起点端
            if not self.start_frontier:
                return False

            f, g, current = heapq.heappop(self.start_frontier)
            self.current_position = current
            self.frontier_set.discard(current)

            # 如果当前节点被终点方向访问，表示相遇
            if current in self.goal_dist and self.goal_dist[current] != float("inf"):
                self.meeting_point = current
                self.build_bidirectional_path()
                return True

            # 如果该节点的 g 值不是最优，跳过
            if g > self.start_dist[current]:
                return False

            self.visited.add(current)  # 记录已访问的节点

            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if self.maze.is_valid(nr, nc):
                    new_g = g + 1
                    if new_g < self.start_dist[(nr, nc)]:
                        self.start_dist[(nr, nc)] = new_g
                        self.start_came_from[(nr, nc)] = current
                        f_new = new_g + self.heuristic((nr, nc), self.goal)
                        heapq.heappush(self.start_frontier, (f_new, new_g, (nr, nc)))
                        self.frontier_set.add((nr, nc))
        else:
            # 扩展终点端
            if not self.goal_frontier:
                return False

            f, g, current = heapq.heappop(self.goal_frontier)
            self.current_position = current
            self.frontier_set.discard(current)

            # 如果当前节点被起点方向访问，表示相遇
            if current in self.start_dist and self.start_dist[current] != float("inf"):
                self.meeting_point = current
                self.build_bidirectional_path()
                return True

            # 如果该节点的 g 值不是最优，跳过
            if g > self.goal_dist[current]:
                return False

            self.visited.add(current)  # 记录已访问的节点

            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if self.maze.is_valid(nr, nc):
                    new_g = g + 1
                    if new_g < self.goal_dist[(nr, nc)]:
                        self.goal_dist[(nr, nc)] = new_g
                        self.goal_came_from[(nr, nc)] = current
                        f_new = new_g + self.heuristic((nr, nc), self.start)
                        heapq.heappush(self.goal_frontier, (f_new, new_g, (nr, nc)))
                        self.frontier_set.add((nr, nc))

        return False
