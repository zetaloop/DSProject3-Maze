import tkinter as tk
from tkinter import ttk
import sv_ttk
import darkdetect
from tkinter import messagebox

from maze import Maze
from solver import AStarSolver


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("迷宫求解可视化演示（A*）")

        # 主题设置
        self.is_dark = darkdetect.isDark()
        if self.is_dark:
            sv_ttk.use_dark_theme()
        else:
            sv_ttk.use_light_theme()

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
        # 创建主控制框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 上方按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # 左侧按钮组
        left_button_frame = ttk.Frame(control_frame)
        left_button_frame.pack(side=tk.LEFT)

        generate_button = ttk.Button(
            left_button_frame, text="生成新迷宫", command=self.on_generate_maze
        )
        generate_button.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(
            left_button_frame, text="开始求解", command=self.on_start_solving
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        reset_button = ttk.Button(left_button_frame, text="重置", command=self.on_reset)
        reset_button.pack(side=tk.LEFT, padx=5)

        # 右侧控制组
        right_control_frame = ttk.Frame(control_frame)
        right_control_frame.pack(side=tk.RIGHT)

        # 主题切换按钮
        theme_button = ttk.Button(
            right_control_frame, text="切换主题", command=self.toggle_theme
        )
        theme_button.pack(side=tk.RIGHT, padx=5)

        # 速度控制组
        speed_frame = ttk.LabelFrame(main_frame, text="动画速度")
        speed_frame.pack(fill=tk.X, padx=5, pady=5)

        self.speed_var = tk.IntVar(value=70)  # 修改默认值
        speed_scale = ttk.Scale(
            speed_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
        )
        speed_scale.pack(fill=tk.X, padx=10, pady=5)

        # 添加速度说明标签
        speed_label = ttk.Label(speed_frame, text="向右拖动提高速度", font=("", 8))
        speed_label.pack(pady=(0, 5))

    def create_canvas(self):
        """
        创建绘制迷宫的画布
        """
        # 创建带滚动条的画布框架
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 添加滚动条
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        # 计算画布大小
        width = self.maze.cols * self.cell_size
        height = self.maze.rows * self.cell_size

        self.canvas = tk.Canvas(
            canvas_frame,
            width=min(width, 800),  # 限制最大显示尺寸
            height=min(height, 600),
            bg="white",
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set,
        )

        # 配置滚动条
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # 放置组件
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 设置画布滚动区域
        self.canvas.config(scrollregion=(0, 0, width, height))

        self.draw_maze()

    def draw_maze(self):
        """绘制迷宫"""
        self.canvas.delete("all")

        # 绘制背景网格
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # 绘制单元格
                if (r, c) == self.maze.start:
                    self.draw_cell(x1, y1, x2, y2, "#00FF66", "起点")
                elif (r, c) == self.maze.goal:
                    self.draw_cell(x1, y1, x2, y2, "#FF0066", "终点")
                elif self.maze.grid[r][c] == 1:
                    self.draw_wall(x1, y1, x2, y2)
                else:
                    self.draw_cell(x1, y1, x2, y2, "#FFFFFF")

        # 绘制搜索过程
        if self.solver:
            # 绘制已访问节点
            for r, c in self.solver.visited:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.draw_cell(x1, y1, x2, y2, "#E1F5FE")  # 淡蓝色

            # 绘制边界节点
            for r, c in self.solver.frontier_set:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.draw_cell(x1, y1, x2, y2, "#FFF3E0")  # 淡橙色

            # 绘制路径
            if self.solver.path:
                for i, (r, c) in enumerate(self.solver.path):
                    if (r, c) not in (self.maze.start, self.maze.goal):
                        x1 = c * self.cell_size
                        y1 = r * self.cell_size
                        x2 = x1 + self.cell_size
                        y2 = y1 + self.cell_size
                        self.draw_path_cell(x1, y1, x2, y2, i, len(self.solver.path))

    def draw_cell(self, x1, y1, x2, y2, color, text=None):
        """绘制一个单元格"""
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=color, outline="#E0E0E0", width=1
        )
        if text:
            self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2, text=text, font=("Arial", 8)
            )

    def draw_wall(self, x1, y1, x2, y2):
        """绘制墙体"""
        # 使用渐变色效果
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill="#424242", outline="#212121", width=1
        )

    def draw_path_cell(self, x1, y1, x2, y2, index, total):
        """绘制路径单元格，使用渐变色效果"""
        # 计算渐变色 - 从绿色渐变到红色
        progress = index / total
        r = int(255 * progress)  # 红色从0渐变到255
        g = int(255 * (1 - progress))  # 绿色从255渐变到0
        color = f"#{r:02x}{g:02x}66"

        # 绘制圆角矩形
        self.canvas.create_rectangle(
            x1 + 2, y1 + 2, x2 - 2, y2 - 2, fill=color, outline="", width=0
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
            if self.solver.path:  # 如果找到路径
                messagebox.showinfo("完成", "已找到最短路径！")
            else:
                messagebox.showwarning("提示", "无法找到可行路径。")
            return

        # 未完成，继续下一步
        # 速度值反转：100 -> 快速（短延迟）, 1 -> 慢速（长延迟）
        delay = int(1000 / self.speed_var.get())  # 将1-100的值转换为延迟时间
        self.after_id = self.root.after(delay, self.solve_step)

    def on_reset(self):
        """
        重置迷宫状态，不重新生成迷宫
        """
        self.stop_solving()
        self.solver = AStarSolver(self.maze)
        self.draw_maze()

    def toggle_theme(self):
        """切换明暗主题"""
        if self.is_dark:
            sv_ttk.use_light_theme()
        else:
            sv_ttk.use_dark_theme()
        self.is_dark = not self.is_dark

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
