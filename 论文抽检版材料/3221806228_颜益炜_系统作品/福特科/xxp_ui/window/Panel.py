from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QRect, QEasingCurve, QPropertyAnimation, Qt

from .HistoryCard import HistoryCard

class ScrollableContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)  # 设置控件间的固定距离

        # 添加一个不可见的空间占位符到布局底部
        spacer_item = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer_item)
        self.layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(self.layout)

    def addWidget(self, widget):
        self.layout.insertWidget(self.layout.count() - 1, widget)  # 在最后一个空间占位符之前插入新控件


class SlidingPanel(QWidget):
    # myDatabase = DBController()
    cards = []
    def __init__(self, parent=None,database=None):
        super(SlidingPanel, self).__init__(parent)
        self.myDatabase=database

        self.setFixedWidth(380)  # 设置侧边栏宽度

        # 移除侧边栏的窗口边框
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_layout = QVBoxLayout(self)
        main_frame = QFrame()
        main_layout.addWidget(main_frame)
        main_frame_layout = QVBoxLayout(main_frame)

        # # 创建并添加历史记录标题
        history_title = QLabel('历史记录', self)
        history_title.setAlignment(Qt.AlignCenter)
        history_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;border:none")
        main_frame_layout.addWidget(history_title)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 始终显示垂直滚动条
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.handle_scroll)
        #
        # # 将内容容器添加到滚动区域
        self.container = ScrollableContainer()
        # self.container.layout.addStretch()
        # self.container.layout.addStretch()
        self.scroll_area.setWidget(self.container)

        # self.grid_layout = QGridLayout(self.scroll_area)
        self.readHistory()

        # 自定义滚动区域样式，使之符合扁平化设计要求，并使滚动条在内部
        self.scroll_area.setStyleSheet("""
            /* 圆角滚动区域 */
            /* 滚动条样式 */
            QScrollBar:vertical {
                width: 12px; /* 调整滚动条宽度 */
                margin: 0px 3px 0px 3px; /* 给滚动条留出一点空间 */
                border-radius: 6px; /* 滚动条圆角 */
            }
            QScrollBar::handle:vertical {
                background: #888; /* 更明显的滚动块颜色 */
                min-height: 20px;
                border-radius: 6px; /* 滚动块圆角 */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        # # 使用一个框架来容纳滚动区域，并设置内边距使得滚动条看起来是在滚动区域内
        container_frame = QFrame()
        container_frame.setStyleSheet("border: none; padding: 0px;")  # 内边距让滚动条看起来在内部
        container_layout = QVBoxLayout(container_frame)
        container_layout.addWidget(self.scroll_area)

        # # 将容器框架添加到主布局
        main_frame_layout.addWidget(container_frame)

        # 设置窗口的布局
        # self.setLayout(main_layout)

        # 应用样式表以创建卡片外观
        self.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 2px solid gray;
        """)
        # self.setGraphicsEffect(shadow)  # 将阴影效果应用到侧边栏上

        # 初始化动画对象
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)  # 动画持续时间
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

    def show_animation(self, main_window_rect):
        """显示侧边栏动画"""
        # print("正在显示侧边栏...")
        start = QRect(main_window_rect.x() + main_window_rect.width() - self.width(), main_window_rect.y(),
                      self.width(), main_window_rect.height())  # 开始于主窗口右侧内侧
        end = QRect(main_window_rect.x() + main_window_rect.width(), main_window_rect.y(), self.width(),
                    main_window_rect.height())  # 结束于主窗口右侧外侧

        self.move(start.topLeft())
        self.show()

        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()

    def hide_animation(self, main_window_rect):
        """隐藏侧边栏动画"""
        # print("正在隐藏侧边栏...")
        start = QRect(main_window_rect.x() + main_window_rect.width(), main_window_rect.y(), self.width(),
                      main_window_rect.height())  # 开始于主窗口右侧外侧
        end = QRect(main_window_rect.x() + main_window_rect.width() - self.width(), main_window_rect.y(), self.width(),
                    main_window_rect.height())  # 结束于主窗口右侧内侧

        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.finished.connect(self.on_animation_finished)
        self.animation.start()

    def on_animation_finished(self):
        """动画完成后的处理"""
        # print("动画已完成，隐藏侧边栏.")
        self.hide()
        try:
            self.animation.finished.disconnect(self.on_animation_finished)
        except TypeError:
            pass  # 如果没有连接则忽略错误

    def handle_scroll(self, action):
        if action >= self.scroll_area.verticalScrollBar().maximum() - 10:
            # print('bottom')
            self.readHistory()

    def clear_child_widgets(self):
        # 清空所有 QPushButton 类型的子控件
        self.remove_child_widgets_by_type(HistoryCard)

    def remove_child_widgets_by_type(self, widget_type):
        # 遍历所有子控件
        for i in reversed(range(self.container.layout.count())):
            widget = self.container.layout.itemAt(i).widget()
            if isinstance(widget, widget_type):
                # 从布局中移除控件
                self.container.layout.removeWidget(widget)
                # 删除控件对象
                widget.deleteLater()

    def readHistory(self):
        # new_card = HistoryCard(parent=self.ui.scrollArea_2, _window=self)
        # self.history_layout.addWidget(new_card)
        historyInfo, begin, show_num = self.myDatabase.getTrayInfo()
        # print(str(historyInfo[0]['Detect_time']))
        # rows, cols = 0 + begin // 2, 0
        for index, item in enumerate(historyInfo):
            print(item['Tray_id'])
            sampleInfo = self.myDatabase.getSampleBelongToTray(item['Tray_id'])

            sampleQuality = []
            isGood = True
            for sample in sampleInfo:
                sampleQuality.append(int(sample['quality']))
                if int(sample['quality']) == 1:
                    isGood = False
            data = {
                'date': str(item['Detect_time']).split(' ')[0],
                'time': str(item['Detect_time']).split(' ')[1],
                'sample': sampleQuality,
                'tip': f'盘编号：{item["Tray_id"]}'
            }
            if isGood:
                data['tip2'] = '不合格'
            else:
                data['tip2'] = '合格'
            # print(data)
            newcard = HistoryCard(_window=self.parent(),data=data)
            self.cards.append(newcard)
            # self.grid_layout.addWidget(newcard)
            # self.grid_layout.update()
            self.container.addWidget(newcard)

    def addNewTrayHistory(self,tray_id):
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        item = self.myDatabase.getOneTray(tray_id)
        sampleInfo = self.myDatabase.getSampleBelongToTray(item['Tray_id'])
        sampleQuality = []
        isGood = True
        for sample in sampleInfo:
            sampleQuality.append(int(sample['quality']))
            if int(sample['quality']) == 1:
                isGood = False
        data = {
            'date': str(item['Detect_time']).split(' ')[0],
            'time': str(item['Detect_time']).split(' ')[1],
            'sample': sampleQuality,
            'tip': f'盘编号：{item["Tray_id"]}'
        }
        if isGood:
            data['tip2'] = '不合格'
        else:
            data['tip2'] = '合格'
        print(data)
        newcard = HistoryCard(_window=self.parent(), data=data)
        self.cards.append(newcard)
        # self.grid_layout.addWidget(newcard)
        # self.grid_layout.update()
        self.container.layout.insertWidget(0,newcard)
        # for card in self.cards:
        #     print(card.geometry())
        #     # self.grid_layout.addWidget(newcard)
        #     # return newcard

