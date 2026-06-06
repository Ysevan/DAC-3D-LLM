import datetime
import json

import matplotlib
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFrame
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from window.Button import Buttons

plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.use("Qt5Agg")  # 声明使用QT5


class GetInfo(QThread):
    message_received = pyqtSignal(dict)  # 定义信号

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._running = True

    def run(self):
        while self._running:
            try:
                msg = self.queue.get()
                if msg:
                    self.message_received.emit(msg)
            except Exception as e:
                # print(f"Worker error: {e}")
                break


def getDateAndTime():
    date = str(datetime.datetime.today().date())
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    second = now.second
    if second < 10:
        second = "0" + str(second)
    if hour < 10:
        hour = "0" + str(hour)
    if minute < 10:
        minute = "0" + str(minute)
    now_time = "{}:{}:{}".format(hour, minute, second)
    # print(date, now_time)
    return date + " " + now_time, date, now_time


# class ButtonManager:
#     def __init__(self,parent):
#         self.buttons = [QPushButton(f'{i + 1}') for i in range(144)]
#         self.initUI(parent)
#
#     def initUI(self,parent):
#         frames = parent.findChildren(QFrame)
#         parent.layout().addWidget(frames[0], 0, 0)
#         parent.layout().addWidget(frames[1], 1, 0)
#         parent.layout().addWidget(frames[2], 0, 1)
#         parent.layout().addWidget(frames[3], 1, 1)
#         for frame in frames:
#             frame.setStyleSheet("border: 2px solid black;")
#             sub_layout = QGridLayout(frame)
#             sub_layout.setSpacing(5)  # 设置按钮之间的间距
#             frame.setLayout(sub_layout)
#         # #
#         # # # 创建总共144个按钮（4个子框架，每个子框架36个按钮）
#
#         # # # 设置按钮样式为圆形
#         for button in self.buttons:
#             button.setFixedSize(QSize(36, 36))  # 设置按钮大小为30x30像素
#             button.setStyleSheet("""
#                        QPushButton {
#                            border-radius: 18px;  /* 设置按钮为圆形 */
#                            background-color: lightgray;
#                            font-size: 15px "黑体";
#                        }
#                        QPushButton:hover {
#                            background-color: gray;  /* 悬停时改变背景颜色 */
#                        }
#                    """)
#         # frames[0].layout().addWidget(buttons.pop(0), 0, 0)
#         button_num = 0
#         for i in range(3):
#             for frame_index in range(2):
#                 for row in range(6):
#                         frames[frame_index].layout().addWidget(self.buttons[button_num], row, i * 2)
#                         button_num += 1
#             for frame_index in reversed(range(2)):
#                 for row in reversed(range(6)):
#                         frames[frame_index].layout().addWidget(self.buttons[button_num], row, i * 2 + 1)
#                         button_num += 1
#         for i in range(3):
#             for frame_index in range(2):
#                 for row in range(6):
#                         frames[frame_index + 2].layout().addWidget(self.buttons[button_num], row, i * 2)
#                         button_num += 1
#             for frame_index in reversed(range(2)):
#                 for row in reversed(range(6)):
#                         frames[frame_index + 2].layout().addWidget(self.buttons[button_num], row, i * 2 + 1)
#                         button_num += 1
#推送
def initButtons(window,frame,size,button_name):
    main_frame = QFrame(frame)
    main_layout = QVBoxLayout()
    main_layout.addWidget(main_frame)
    buttons = Buttons(window,main_frame, size,button_name)
    frame.setLayout(main_layout)
    return buttons



def find_button(parent):
    buttonList = []
    for button in parent.findChildren(QPushButton):
        buttonList.append(button)
    return buttonList

def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(data)


def sendDataToZYC(data):
    # 使用示例
    print('向服务端发送：',data)
    filename = r'../msg/shared_data_win_to_ZYC.txt'
    # data = '''
    #             name: Alice,
    #             age: 30,
    #             check: true
    #     '''
    write_to_file(filename, data)


def convert_nested_dict_to_list(d):
    if isinstance(d, dict):
        # print(d.keys())
        for key, value in d.items():
            # 如果值是字典且键都是数字
            if isinstance(value, dict) and all(k.isdigit() for k in value):
                # 转换为列表
                # print(value)
                if isinstance(value, dict):
                    convert_nested_dict_to_list(value)
                d[key] = list(value.values())
            # 如果值是字典，递归调用自身
            elif isinstance(value, dict):
                convert_nested_dict_to_list(value)
        if all(k.isdigit() for k in d.keys()):
            d = list(d.values())
    return d


def receiveDataFromZYC():
    # 使用示例
    filename = r'../msg/shared_data_ZYC_to_win.txt'
    with open(filename, 'r+', encoding='utf-8') as file:
        content = file.read()
        # print(content)
        # 尝试解析 JSON 字符串
        if len(content) > 0:
            try:
                # 解析 JSON 字符串并将其转换为 Python 对象
                data = json.loads(content)
                # print(data)
                data = convert_nested_dict_to_list(data)

                # 文件指针移到文件开头
                file.seek(0)
                # 将文件指针移到文件末尾
                file.truncate()
                return data
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                print(content)

class VariableMonitor(QObject):
    # 定义一个信号，当变量变化时发出
    variable_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._variable = False

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, value):
        if self._variable != value:
            self._variable = value
            # 当变量变化时，发出信号
            self.variable_changed.emit(self._variable)
class FigBox(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        dynamic_canvas = FigureCanvas(Figure(figsize=(10, 5)))
        layout.addWidget(dynamic_canvas)
        self._dynamic_ax = dynamic_canvas.figure.subplots()

        # self.plot()

    def plot(self, msg):
        # 示例数据
        # data = {'A': 5, 'B': 7, 'C': 3, 'D': 2, 'E': 4}

        data = msg['data']

        self._dynamic_ax.clear()
        p1 = self._dynamic_ax.bar(data.keys(), data.values(), width=0.4)
        # self.axes.set_title(results['title'])
        self._dynamic_ax.bar_label(p1)
        self._dynamic_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self._dynamic_ax.set_title(msg['title'])
        y_ticks = self._dynamic_ax.get_yticks()
        self._dynamic_ax.set_yticks(y_ticks)
        if msg['title'] == '无瑕疵':
            self._dynamic_ax.set_yticklabels(['' for label in self._dynamic_ax.get_yticklabels()])
        # self._dynamic_ax.set_xlabel('类别')
        # self._dynamic_ax.set_ylabel('值')
        self._dynamic_ax.figure.canvas.draw()


def plotFig(parent, data, figBox):
    layout = QVBoxLayout()
    layout.addWidget(figBox)
    parent.setLayout(layout)
    figBox.plot(data)


# def plotFig(data, ui, groupbox):
#     F = MyFigure(width=10, height=10, dpi=90)
#     F.drawFig(data)
#     ui.gridlayout = QGridLayout(groupbox)  # 继承容器groupBox
#     ui.gridlayout.addWidget(F, 0, 1)


def changeSampleState(button, state, historyMode=False):
    if state == 1:
        r1 = 'rgb(0,200,0)'
        r2 = 'rgb(0,150,0)'
    elif state == 0:
        r1 = 'rgb(200,0,0)'
        r2 = 'rgb(150,0,0)'
    else:
        r1 = 'rgb(200,200,200)'
        r2 = 'rgb(150,150,150)'
    if not historyMode:
        button.setStyleSheet(
            '''
            QPushButton{
                border:1px solid black;
                border-radius:15px;
                background-color: %s;
            }
            QPushButton:hover{
                border:1px solid black;
                border-radius:15px;
                background-color: %s;
            }
            ''' % (r1, r2)
        )
    else:
        button.setStyleSheet(
            '''
            QPushButton{
                border:1px solid black;
                border-radius:5px;
                background-color: %s;
            }
            QPushButton:hover{
                border:1px solid black;
                border-radius:5px;
                background-color: %s;
            }
            ''' % (r1, r2)
        )


if __name__ == '__main__':
    receiveDataFromZYC()
