import tkinter as tk
from tkinter import ttk
import sv_ttk

from maze import Maze
from solver import AStarSolver


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("迷宫求解可视化演示（A*）")

        # 使用sv_ttk主题
        sv_ttk.use_dark_theme()

        # 初始化界面布局
        self.setup_ui()

        # 初始化迷宫对象和求解算法
        self.maze_size = 20  # 可以根据需要调整迷宫大小
        self.maze = Maze(self.maze_size, self.maze_size)
        self.cell_size = 25

        self.solver = AStarSolver(self.maze)
        self.is_solving = False
        self.after_id = None

        # 创建迷宫画布
        self.create_canvas()

    def setup_ui(self):
        """
        创建顶部控制区域、速度调节、按钮等
        """
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 生成迷宫按钮
        generate_button = ttk.Button(
            control_frame, text="生成新迷宫", command=self.on_generate_maze
        )
        generate_button.pack(side=tk.LEFT, padx=5)

        # 开始求解按钮
        self.start_button = ttk.Button(
            control_frame, text="开始求解", command=self.on_start_solving
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # 重置按钮
        reset_button = ttk.Button(control_frame, text="重置", command=self.on_reset)
        reset_button.pack(side=tk.LEFT, padx=5)

        # 速度调节
        speed_label = ttk.Label(control_frame, text="可视化速度(ms):")
        speed_label.pack(side=tk.LEFT, padx=5)

        self.speed_var = tk.IntVar(value=200)
        speed_scale = ttk.Scale(
            control_frame,
            from_=50,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self.on_speed_change,
            variable=self.speed_var,
        )
        speed_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def create_canvas(self):
        """
        创建绘制迷宫的画布
        """
        width = self.maze.cols * self.cell_size
        height = self.maze.rows * self.cell_size

        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)

        self.draw_maze()

    def draw_maze(self):
        """
        根据 maze 对象的 wall 信息绘制迷宫格子、起点终点等
        """
        self.canvas.delete("all")  # 清空画布

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if (r, c) == self.maze.start:
                    color = "green"
                elif (r, c) == self.maze.goal:
                    color = "red"
                elif self.maze.grid[r][c] == 1:
                    color = "black"  # 墙体
                else:
                    color = "white"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        # 绘制搜索过程中标记的颜色
        if self.solver:
            for r, c in self.solver.visited:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="#87CEFA", outline="gray"  # 浅蓝色
                    )
            for r, c in self.solver.frontier_set:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="#F0E68C", outline="gray"  # 浅卡其色
                    )

            # 最终找到的路径
            for r, c in self.solver.path:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="#FFA500", outline="gray"  # 橙色
                    )

    def on_generate_maze(self):
        """
        点击“生成新迷宫”按钮后回调，重新生成迷宫
        """
        self.stop_solving()
        self.maze.generate_random_maze()
        self.solver = AStarSolver(self.maze)
        self.draw_maze()

    def on_start_solving(self):
        """
        点击“开始求解”按钮后回调
        """
        if self.is_solving:
            return

        self.is_solving = True
        self.solver.reset()
        self.solve_step()

    def solve_step(self):
        """
        逐步执行求解算法，并在界面上刷新状态
        """
        done = self.solver.step()
        self.draw_maze()

        if done:
            self.is_solving = False
            return

        # 未完成，继续下一步
        self.after_id = self.root.after(self.speed_var.get(), self.solve_step)

    def on_reset(self):
        """
        重置迷宫状态，不重新生成迷宫
        """
        self.stop_solving()
        self.solver = AStarSolver(self.maze)
        self.draw_maze()

    def on_speed_change(self, event):
        """
        速度调节回调
        """
        pass  # self.speed_var.get() 即可取到当前滑动条值

    def stop_solving(self):
        """
        如果正在求解，停止定时器
        """
        self.is_solving = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None


def main():
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
