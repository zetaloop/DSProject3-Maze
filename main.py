import tkinter as tk
from tkinter import ttk
import sv_ttk
import darkdetect
from tkinter import messagebox

from maze import Maze
from solver import (
    DFSSolver,
    BFSSolver,
    BidirectionalBFSSolver,
    GreedySolver,
    AStarSolver,
)


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("迷宫求解可视化演示")

        # 主题设置
        self.is_dark = darkdetect.isDark()
        if self.is_dark:
            sv_ttk.use_dark_theme()
        else:
            sv_ttk.use_light_theme()

        # 设置默认字体
        self.default_font = ("等线", 10)
        self.small_font = ("等线", 8)

        # 双倍大小模式
        self.double_size_mode = tk.BooleanVar(value=False)
        self.base_maze_size = 19  # 基础迷宫大小
        self.base_cell_size = 30  # 基础格子大小

        # 迷宫类型字典
        self.maze_types = {
            "空白": (
                "generate_empty_maze",
                "完全空白的地形，适合纯粹展示算法的搜索过程",
            ),
            "障碍": (
                "generate_random_maze",
                "随机分布的障碍物，适合一般寻路测试",
            ),
            "迷宫": (
                "generate_traditional_maze",
                "经典的迷宫布局",
            ),
            "山丘": ("generate_block_maze", "大块的障碍物分布，模拟真实山丘地形"),
            "河流": (
                "generate_river_maze",
                "路径曲折的河流地形",
            ),
        }
        self.current_maze_type = "空白"
        self.ensure_path = tk.BooleanVar(value=True)  # 添加确保路径存在的变量

        # 初始化迷宫对象和求解算法
        self.maze_size = self.base_maze_size  # 修改为奇数，这样传统迷宫算法可以正确工作
        self.maze = Maze(self.maze_size, self.maze_size)
        self.cell_size = self.base_cell_size

        # 算法字典
        self.solver_classes = {
            "深度优先搜索 (DFS)": DFSSolver,
            "广度优先搜索 (BFS)": BFSSolver,
            "双向广度优先搜索": BidirectionalBFSSolver,
            "贪心搜索": GreedySolver,
            "A*搜索": AStarSolver,
        }

        # 算法说明
        self.algorithm_descriptions = {
            "深度优先搜索 (DFS)": "递归探索一条路径直到无法继续，内存占用小但不保证最短路径",
            "广度优先搜索 (BFS)": "按层次扩展搜索，在无权图中保证找到最短路径",
            "双向广度优先搜索": "同时从起点和终点搜索，在大型迷宫中通常更高效",
            "贪心搜索": "仅使用启发式估计选择下一步，速度快但不保证最短路径",
            "A*搜索": "结合实际代价和启发式估计，通常能找到最短路径",
        }

        self.current_algorithm = "深度优先搜索 (DFS)"
        self.solver = self.solver_classes[self.current_algorithm](self.maze)
        self.is_solving = False
        self.after_id = None

        # 初始化界面布局
        self.setup_ui()

        # 创建迷宫画布
        self.create_canvas()

    def setup_ui(self):
        """创建顶部控制区域、速度调节、按钮等"""
        # 创建主控制框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 迷宫类型选择区域
        maze_frame = ttk.LabelFrame(main_frame, text="迷宫类型")
        maze_frame.pack(fill=tk.X, padx=5, pady=5)

        maze_combo_frame = ttk.Frame(maze_frame)
        maze_combo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(maze_combo_frame, text="选择地形：").pack(side=tk.LEFT)
        self.maze_type_var = tk.StringVar(value=self.current_maze_type)
        maze_combo = ttk.Combobox(
            maze_combo_frame,
            textvariable=self.maze_type_var,
            values=list(self.maze_types.keys()),
            state="readonly",
            width=25,
        )
        maze_combo.pack(side=tk.LEFT, padx=5)
        maze_combo.bind("<<ComboboxSelected>>", self.on_maze_type_changed)

        # 添加确保路径存在的开关
        ensure_path_check = ttk.Checkbutton(
            maze_combo_frame,
            text="确保可达终点",
            variable=self.ensure_path,
            command=self.on_ensure_path_changed,
        )
        ensure_path_check.pack(side=tk.RIGHT, padx=5)

        # 迷宫类型描述标签
        self.maze_description_label = ttk.Label(
            maze_frame,
            text=self.maze_types[self.current_maze_type][1],
            wraplength=600,
            justify=tk.LEFT,
            font=self.small_font,
        )
        self.maze_description_label.pack(fill=tk.X, padx=5, pady=5)

        # 算法选择区域
        algorithm_frame = ttk.LabelFrame(main_frame, text="算法选择")
        algorithm_frame.pack(fill=tk.X, padx=5, pady=5)

        # 算法选择下拉框
        combo_frame = ttk.Frame(algorithm_frame)
        combo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(combo_frame, text="选择算法：").pack(side=tk.LEFT)
        self.algorithm_var = tk.StringVar(value=self.current_algorithm)
        algorithm_combo = ttk.Combobox(
            combo_frame,
            textvariable=self.algorithm_var,
            values=list(self.solver_classes.keys()),
            state="readonly",
            width=25,
        )
        algorithm_combo.pack(side=tk.LEFT, padx=5)
        algorithm_combo.bind("<<ComboboxSelected>>", self.on_algorithm_changed)

        # 算法描述标签
        self.description_label = ttk.Label(
            algorithm_frame,
            text=self.algorithm_descriptions[self.current_algorithm],
            wraplength=600,
            justify=tk.LEFT,
            font=self.small_font,
        )
        self.description_label.pack(fill=tk.X, padx=5, pady=5)

        # 速度控制组
        speed_frame = ttk.LabelFrame(main_frame, text="动画速度")
        speed_frame.pack(fill=tk.X, padx=5, pady=5)

        speed_control_frame = ttk.Frame(speed_frame)
        speed_control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.speed_var = tk.IntVar(value=80)
        speed_scale = ttk.Scale(
            speed_control_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.on_speed_changed,  # 添加回调函数
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 添加显示具体数值的标签
        self.speed_label = ttk.Label(
            speed_control_frame,
            text="80",  # 初始值
            width=3,
            anchor=tk.E,
        )
        self.speed_label.pack(side=tk.LEFT, padx=(5, 0))

        speed_hint = ttk.Label(
            speed_frame,
            text="向右拖动提高速度（1-100）",
            font=self.small_font,
        )
        speed_hint.pack(pady=(0, 5))

        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # 左侧按钮组
        left_button_frame = ttk.Frame(control_frame)
        left_button_frame.pack(side=tk.LEFT)

        generate_button = ttk.Button(
            left_button_frame, text="生成新迷宫", command=self.on_generate_maze
        )
        generate_button.pack(side=tk.LEFT, padx=5)
        generate_button.configure(style="Custom.TButton")

        self.start_button = ttk.Button(
            left_button_frame, text="开始求解", command=self.on_start_solving
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.start_button.configure(style="Custom.TButton")

        reset_button = ttk.Button(left_button_frame, text="重置", command=self.on_reset)
        reset_button.pack(side=tk.LEFT, padx=5)
        reset_button.configure(style="Custom.TButton")

        # 右侧按钮组
        right_button_frame = ttk.Frame(control_frame)
        right_button_frame.pack(side=tk.RIGHT)

        # 双倍大小模式切换按钮
        double_size_check = ttk.Checkbutton(
            right_button_frame,
            text="双倍大小",
            variable=self.double_size_mode,
            command=self.on_size_mode_changed,
        )
        double_size_check.pack(side=tk.RIGHT, padx=5)

        # 主题切换按钮
        theme_button = ttk.Button(
            right_button_frame, text="☀", command=self.toggle_theme
        )
        theme_button.pack(side=tk.RIGHT, padx=5)
        theme_button.configure(style="Custom.TButton")

    def create_canvas(self):
        """创建绘制迷宫的画布"""
        # 创建画布框架
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 计算画布大小
        width = self.maze.cols * self.cell_size
        height = self.maze.rows * self.cell_size

        # 创建画布
        self.canvas = tk.Canvas(
            canvas_frame,
            width=width,
            height=height,
            bg="white",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

        # 存储上一次的状态，用于增量更新
        self._last_visited = set()
        self._last_frontier = set()
        self._last_path = []
        self._last_position = None

    def update_maze_view(self):
        """增量更新迷宫视图"""
        if not self.solver:
            return

        # 更新已访问节点
        new_visited = self.solver.visited - self._last_visited
        for r, c in new_visited:
            if (r, c) not in (self.maze.start, self.maze.goal):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.draw_cell(x1, y1, x2, y2, "#E1F5FE")  # 淡蓝色

        # 更新边界节点
        old_frontier = self._last_frontier - self.solver.frontier_set
        new_frontier = self.solver.frontier_set - self._last_frontier

        # 清除旧的边界节点（恢复为白色或已访问的颜色）
        for r, c in old_frontier:
            if (r, c) not in (self.maze.start, self.maze.goal):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "#E1F5FE" if (r, c) in self.solver.visited else "#FFFFFF"
                self.draw_cell(x1, y1, x2, y2, color)

        # 绘制新的边界节点
        for r, c in new_frontier:
            if (r, c) not in (self.maze.start, self.maze.goal):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.draw_cell(x1, y1, x2, y2, "#FFF3E0")  # 淡橙色

        # 更新路径
        if self.solver.path != self._last_path:
            # 清除旧路径
            for r, c in self._last_path:
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    color = "#E1F5FE" if (r, c) in self.solver.visited else "#FFFFFF"
                    self.draw_cell(x1, y1, x2, y2, color)

            # 绘制新路径
            for i, (r, c) in enumerate(self.solver.path):
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.draw_path_cell(x1, y1, x2, y2, i, len(self.solver.path))

        # 更新当前位置
        if self.solver.current_position != self._last_position:
            # 清除旧的当前位置
            if self._last_position:
                r, c = self._last_position
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    # 根据节点状态决定颜色
                    if (r, c) in self.solver.path:
                        i = self.solver.path.index((r, c))
                        self.draw_path_cell(x1, y1, x2, y2, i, len(self.solver.path))
                    elif (r, c) in self.solver.frontier_set:
                        self.draw_cell(x1, y1, x2, y2, "#FFF3E0")
                    elif (r, c) in self.solver.visited:
                        self.draw_cell(x1, y1, x2, y2, "#E1F5FE")
                    else:
                        self.draw_cell(x1, y1, x2, y2, "#FFFFFF")

            # 绘制新的当前位置
            if self.solver.current_position:
                r, c = self.solver.current_position
                if (r, c) not in (self.maze.start, self.maze.goal):
                    x1 = c * self.cell_size
                    y1 = r * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.draw_current_position(x1, y1, x2, y2)

        # 更新状态
        self._last_visited = self.solver.visited.copy()
        self._last_frontier = self.solver.frontier_set.copy()
        self._last_path = self.solver.path.copy() if self.solver.path else []
        self._last_position = self.solver.current_position

    def draw_cell(self, x1, y1, x2, y2, color, text=None):
        """绘制一个单元格"""
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=color, outline="#E0E0E0", width=1
        )
        if text:
            self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2, text=text, font=self.small_font
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

    def draw_current_position(self, x1, y1, x2, y2):
        """绘制当前位置，使用醒目的颜色"""
        # 使用深紫色作为当前位置的标记
        self.canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x2 - 2,
            y2 - 2,
            fill="#4FC3F7",  # 中等深度的蓝色
            outline="#039BE5",  # 稍深的蓝色边框
            width=2,
        )
        # 添加一个小圆点作为中心标记
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius = min(self.cell_size / 6, 5)  # 圆点半径，不超过格子的1/6且最大5像素
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill="white",
            outline="",
        )

    def on_generate_maze(self):
        """点击"生成新迷宫"按钮后回调，重新生成迷宫"""
        self.stop_solving()
        # 根据当前选择的迷宫类型生成迷宫
        maze_generator = getattr(self.maze, self.maze_types[self.current_maze_type][0])

        # 对于传统迷宫，不传入ensure_path参数（因为它总是有路径的）
        if self.current_maze_type == "迷宫":
            maze_generator()
        else:
            maze_generator(ensure_path=self.ensure_path.get())

        self.solver = self.solver_classes[self.current_algorithm](self.maze)
        self.draw_maze()

    def on_start_solving(self):
        """
        点击"开始求解"按钮后回调
        """
        if self.is_solving:
            return

        self.is_solving = True
        self.solver.reset()
        # 重新绘制整个迷宫以清除之前的状态
        self.draw_maze()
        self.solve_step()

    def solve_step(self):
        """
        逐步执行求解算法，并在界面上刷新状态
        """
        done = self.solver.step()
        self.update_maze_view()  # 使用增量更新替代完全重绘

        if done:
            self.is_solving = False
            if self.solver.path:  # 如果找到路径
                messagebox.showinfo("完成", "已到达终点！")
            else:
                messagebox.showwarning("提示", "无法找到可行路径。")
            return

        # 未完成，继续下一步
        speed = self.speed_var.get()
        if speed >= 95:  # 最快速度时几乎无延迟
            delay = 1
        else:
            delay = int(1000 * (1.0 - speed / 100.0) ** 2)  # 使用指数衰减
        self.after_id = self.root.after(delay, self.solve_step)

    def on_reset(self):
        """
        重置迷宫状态，不重新生成迷宫
        """
        self.stop_solving()
        self.solver = self.solver_classes[self.current_algorithm](self.maze)
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

    def on_algorithm_changed(self, event):
        """当选择的算法改变时调用"""
        self.current_algorithm = self.algorithm_var.get()
        self.solver = self.solver_classes[self.current_algorithm](self.maze)
        # 更新算法描述
        self.description_label.config(
            text=self.algorithm_descriptions[self.current_algorithm]
        )
        self.on_reset()  # 重置当前状态

    def on_maze_type_changed(self, event):
        """当选择的迷宫类型改变时调用"""
        self.current_maze_type = self.maze_type_var.get()
        # 更新迷宫类型描述
        self.maze_description_label.config(
            text=self.maze_types[self.current_maze_type][1]
        )
        self.on_generate_maze()  # 重新生成迷宫

    def on_ensure_path_changed(self):
        """当确保路径存在的开关状态改变时调用"""
        self.on_generate_maze()  # 重新生成迷宫

    def on_size_mode_changed(self):
        """当双倍大小模式改变时调用"""
        if self.double_size_mode.get():
            self.maze_size = self.base_maze_size * 2 + 1  # 39
            self.cell_size = self.base_cell_size // 2  # 15
        else:
            self.maze_size = self.base_maze_size  # 19
            self.cell_size = self.base_cell_size  # 30

        # 重新创建迷宫
        self.maze = Maze(self.maze_size, self.maze_size)
        self.solver = self.solver_classes[self.current_algorithm](self.maze)

        # 重新调整画布大小
        width = self.maze.cols * self.cell_size
        height = self.maze.rows * self.cell_size
        self.canvas.config(width=width, height=height)

        # 重新生成迷宫
        self.on_generate_maze()

    def on_speed_changed(self, value):
        """当速度滑块值改变时调用"""
        try:
            # 更新速度标签
            self.speed_label.config(text=str(int(float(value))))
        except:
            pass


def main():
    root = tk.Tk()
    # 创建自定义样式
    style = ttk.Style()
    style.configure("Custom.TButton", font=("等线", 10))
    style.configure("TLabelframe.Label", font=("等线", 10))
    app = MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
