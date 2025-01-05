import heapq
import math
from collections import deque


class BaseSolver:
    """所有寻路算法的基类"""

    def __init__(self, maze):
        self.maze = maze
        self.start = maze.start
        self.goal = maze.goal
        self.visited = set()
        self.frontier_set = set()  # 用于界面显示
        self.came_from = {}
        self.path = []

    def reset(self):
        """重置算法状态"""
        self.visited = set()
        self.frontier_set = set()
        self.came_from = {}
        self.path = []

    def build_path(self, current):
        """根据came_from信息构建路径"""
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        self.path = path


class DFSSolver(BaseSolver):
    """深度优先搜索算法"""

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
            return True

        current = self.stack.pop()
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True

        self.visited.add(current)

        # 检查四个方向
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


class BFSSolver(BaseSolver):
    """广度优先搜索算法"""

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
            return True

        current = self.queue.popleft()
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True

        self.visited.add(current)

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


class DijkstraSolver(BaseSolver):
    """Dijkstra算法"""

    def __init__(self, maze):
        super().__init__(maze)
        self.pq = []
        self.reset()

    def reset(self):
        super().reset()
        self.pq = [(0, self.start)]
        self.frontier_set.add(self.start)

    def step(self):
        if not self.pq:
            return True

        cost, current = heapq.heappop(self.pq)
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True

        if current in self.visited:
            return False

        self.visited.add(current)

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in self.visited:
                new_cost = cost + 1
                heapq.heappush(self.pq, (new_cost, (nr, nc)))
                self.frontier_set.add((nr, nc))
                self.came_from[(nr, nc)] = current

        return False


class BidirectionalBFSSolver(BaseSolver):
    """双向广度优先搜索"""

    def __init__(self, maze):
        super().__init__(maze)
        self.start_queue = deque()
        self.goal_queue = deque()
        self.start_visited = set()
        self.goal_visited = set()
        self.start_came_from = {}
        self.goal_came_from = {}
        self.meeting_point = None
        self.reset()

    def reset(self):
        super().reset()
        self.start_queue = deque([self.start])
        self.goal_queue = deque([self.goal])
        self.start_visited = set()
        self.goal_visited = set()
        self.start_came_from = {}
        self.goal_came_from = {}
        self.meeting_point = None
        self.frontier_set = {self.start, self.goal}

    def build_bidirectional_path(self):
        """构建双向搜索的完整路径"""
        # 从起点到相遇点的路径
        path_from_start = []
        current = self.meeting_point
        while current in self.start_came_from:
            path_from_start.append(current)
            current = self.start_came_from[current]
        path_from_start.append(self.start)
        path_from_start.reverse()

        # 从相遇点到终点的路径
        path_to_goal = []
        current = self.meeting_point
        while current in self.goal_came_from:
            current = self.goal_came_from[current]
            path_to_goal.append(current)

        # 合并路径
        self.path = path_from_start + path_to_goal

    def step(self):
        if not self.start_queue or not self.goal_queue:
            return True

        # 从起点方向扩展
        current = self.start_queue.popleft()
        self.frontier_set.discard(current)

        if current in self.goal_visited:
            self.meeting_point = current
            self.build_bidirectional_path()
            return True

        self.start_visited.add(current)

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in self.start_visited:
                self.start_queue.append((nr, nc))
                self.frontier_set.add((nr, nc))
                self.start_came_from[(nr, nc)] = current

        # 从终点方向扩展
        current = self.goal_queue.popleft()
        self.frontier_set.discard(current)

        if current in self.start_visited:
            self.meeting_point = current
            self.build_bidirectional_path()
            return True

        self.goal_visited.add(current)

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in self.goal_visited:
                self.goal_queue.append((nr, nc))
                self.frontier_set.add((nr, nc))
                self.goal_came_from[(nr, nc)] = current

        return False


class AStarSolver(BaseSolver):
    """A*寻路算法"""

    def __init__(self, maze):
        super().__init__(maze)
        self.frontier = []
        self.reset()

    def reset(self):
        super().reset()
        # heapq 需要放入 (优先级, 已走的步数, 节点)
        start_node = (0, 0, self.start)
        heapq.heappush(self.frontier, start_node)
        self.frontier_set.add(self.start)

    def heuristic(self, pos1, pos2):
        """启发式函数 - 曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        if not self.frontier:
            return True

        # 从优先队列中取出优先级最低（f值最小）的节点
        f, cost, current = heapq.heappop(self.frontier)
        self.frontier_set.discard(current)

        if current == self.goal:
            self.build_path(current)
            return True

        self.visited.add(current)

        # 检查周围可行走节点
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if self.maze.is_valid(nr, nc) and (nr, nc) not in self.visited:
                next_cost = cost + 1
                h = self.heuristic((nr, nc), self.goal)
                f_new = next_cost + h
                new_node = (f_new, next_cost, (nr, nc))

                # 如果该邻居没有在frontier中，则加入
                if (nr, nc) not in self.frontier_set:
                    heapq.heappush(self.frontier, new_node)
                    self.frontier_set.add((nr, nc))
                    self.came_from[(nr, nc)] = current

        return False
