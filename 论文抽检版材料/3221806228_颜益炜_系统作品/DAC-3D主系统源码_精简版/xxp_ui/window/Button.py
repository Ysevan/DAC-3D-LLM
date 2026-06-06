import sys
from PyQt5.QtWidgets import QApplication, QFrame, QWidget, QVBoxLayout, QGridLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPalette

def arrangeButtons(frames, buttons):
    button_num = 0
    for i in range(3):
        for frame_index in range(2):
            for row in range(6):
                frames[frame_index].layout().addWidget(buttons[button_num], row, i * 2)
                button_num += 1
        for frame_index in reversed(range(2)):
            for row in reversed(range(6)):
                frames[frame_index].layout().addWidget(buttons[button_num], row, i * 2 + 1)
                button_num += 1
    for i in range(3):
        for frame_index in range(2):
            for row in range(6):
                frames[frame_index + 2].layout().addWidget(buttons[button_num], row, i * 2)
                button_num += 1
        for frame_index in reversed(range(2)):
            for row in reversed(range(6)):
                frames[frame_index + 2].layout().addWidget(buttons[button_num], row, i * 2 + 1)
                button_num += 1


class SingleButton(QPushButton):
    def __init__(self, win,sample_id, status='pending', parent=None, size=30,name='detect'):
        super().__init__(parent)
        self.sample_id = sample_id
        self.status = status
        self.size = size
        self.win = win
        self.setFixedSize(size, size)  # 设置为正方形，以确保是圆形
        self.setObjectName(f'{name}_{sample_id}')
        self.setStyleSheet(self.get_style_sheet(status))
        self.setText(sample_id)  # 设置按钮文本为样品编号
        # self.setToolTip(f"Sample {sample_id} is {status.title()}")
        self.setEnabled(False)  # 禁用按钮，使其不可点击
        self.clicked.connect(lambda: win.XXPResShowBtnClick())

    def get_style_sheet(self, status):
        colors = {
            'pass': ('green', '#006400'),
            'fail': ('red', '#8B0000'),
            'pending': ('gray', '#A9A9A9')
        }
        color, pressed_color = colors[status]
        return f"""
            QPushButton {{
                background-color: {color};
                border: 2px solid #555;
                border-radius: {self.size // 2}px; /* 半径为宽度或高度的一半 */
                padding: 2px;
                color: white;
                font-size: {self.size // 2 - 2}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {pressed_color};
            }}
        """

    def set_style_sheet(self, status):
        self.setStyleSheet(self.get_style_sheet(status))





class Buttons(QWidget):
    def __init__(self, win, parent=None, size=20,button_name='detect'):
        super().__init__(parent)
        self.buttons = [SingleButton(win, str(i + 1), size=size,name=button_name) for i in range(144)]
        self.size = size
        self.window = win
        self.initUI()

    def initUI(self):
        # 创建主框架
        main_frame = QFrame(self)
        main_frame.setStyleSheet("border:2px solid black;")
        # main_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        # main_frame.setLineWidth(2)
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_frame)
        self.setLayout(main_layout)

        # 创建网格布局
        grid_layout = QGridLayout()
        main_frame.setLayout(grid_layout)

        # 添加四个自定义框架到网格布局中
        frames = []
        for i in range(2):
            for j in range(2):
                frame = QFrame(main_frame)
                sub_layout = QGridLayout(frame)
                sub_layout.setSpacing(5)  # 设置按钮之间的间距
                frame.setLayout(sub_layout)
                grid_layout.addWidget(frame, j, i)
                frames.append(frame)

        # 设置拉伸因子（可选），确保子frame随父组件变换大小
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        # for button in self.buttons:
        #     button.setFixedSize(QSize(self.size,self.size)) # 设置按钮大小为30x30像素
        #     button.setStyleSheet(
        #         """
        #                       QPushButton {
        #                           border-radius: %s;  /* 设置按钮为圆形 */
        #                           border:2px solid black;
        #                           font-size: %s "黑体";
        #                       }
        #                       QPushButton:hover {
        #                           background-color: gray;  /* 悬停时改变背景颜色 */
        #                       }
        #         """ % (str(self.size//2)+'px',str(self.size//2-2)+'px')
        #     )
        #     button.setEnabled(False)
        # frames[0].layout().addWidget(buttons.pop(0), 0, 0)

        arrangeButtons(frames,self.buttons)

        # button_num = 0
        # for i in range(3):
        #     for frame_index in range(2):
        #         for row in range(6):
        #             frames[frame_index].layout().addWidget(self.buttons[button_num], row, i * 2)
        #             button_num += 1
        #     for frame_index in reversed(range(2)):
        #         for row in reversed(range(6)):
        #             frames[frame_index].layout().addWidget(self.buttons[button_num], row, i * 2 + 1)
        #             button_num += 1
        # for i in range(3):
        #     for frame_index in range(2):
        #         for row in range(6):
        #             frames[frame_index + 2].layout().addWidget(self.buttons[button_num], row, i * 2)
        #             button_num += 1
        #     for frame_index in reversed(range(2)):
        #         for row in reversed(range(6)):
        #             frames[frame_index + 2].layout().addWidget(self.buttons[button_num], row, i * 2 + 1)
        #             button_num += 1


    def changeButtonType(self, sample_id, status):
        button = self.buttons[sample_id - 1]
        button.set_style_sheet(status)
        if status == 'pending':
            button.setEnabled(False)
        else:
            button.setEnabled(True)

    def changeAllButtonType(self,status):
        for i, button in enumerate(self.buttons):
            button.set_style_sheet(status[i])
            if not status[i] == 'pending':
                button.setEnabled(True)
            else:
                button.setEnabled(False)

    def set_button_result_data(self, sample_id, result_image_path, defections=None):
        """设置指定按钮的结果数据"""
        if 1 <= sample_id <= 144:
            self.buttons[sample_id - 1].set_result_data(result_image_path, defections)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        main_frame = QFrame(self)
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_frame)
        Button = Buttons(main_frame, 36)
        Button.changeButtonType(20, 'fail')
        self.setLayout(main_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
