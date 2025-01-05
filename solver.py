import heapq
import math


class AStarSolver:
    def __init__(self, maze):
        self.maze = maze
        self.start = maze.start
        self.goal = maze.goal

        # 路径、开放列表、已访问列表
        self.frontier = []
        self.frontier_set = set()  # 用于界面显示
        self.visited = set()
        self.came_from = {}
        self.path = []

        self.reset()

    def reset(self):
        """
        重新初始化算法状态
        """
        self.frontier = []
        self.frontier_set = set()
        self.visited = set()
        self.came_from = {}
        self.path = []

        # heapq 需要放入 (优先级, 已走的步数, 节点)
        start_node = (0, 0, self.start)
        heapq.heappush(self.frontier, start_node)
        self.frontier_set.add(self.start)

    def heuristic(self, pos1, pos2):
        """
        启发式函数 - 曼哈顿距离
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        """
        让算法运行一步。如果已经到达目标或无路可走，返回 True，否则 False。
        """
        if not self.frontier:
            # 无路可走
            return True

        # 从优先队列中取出优先级最低（f值最小）的节点
        f, cost, current = heapq.heappop(self.frontier)
        self.frontier_set.discard(current)

        # 如果已经到达目标
        if current == self.goal:
            self.build_path(current)
            return True

        # 将当前节点标记为已访问
        self.visited.add(current)

        # 检查周围可行走节点
        for delta_r, delta_c in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = current[0] + delta_r, current[1] + delta_c
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
                else:
                    # 如果已经在 frontier 中，一般需要比较 cost
                    # 此示例为了简化，暂不做重复节点的处理
                    pass

        return False

    def build_path(self, current):
        """
        根据 came_from 信息回溯路径
        """
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        self.path = path
