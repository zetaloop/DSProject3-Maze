import random


class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0] * cols for _ in range(rows)]

        # 起点和目标位置可根据需求随机生成或指定
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)

        self.generate_random_maze()

    def generate_random_maze(self, wall_ratio=0.3):
        """
        通过随机的方法生成迷宫墙体，wall_ratio 为随机墙的占比
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # 设置边界
                if (r, c) == self.start or (r, c) == self.goal:
                    self.grid[r][c] = 0
                else:
                    self.grid[r][c] = 1 if random.random() < wall_ratio else 0

    def is_valid(self, r, c):
        """
        判断是否在迷宫范围内且不是墙
        """
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 0
        return False
