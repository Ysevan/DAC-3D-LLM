from PyQt5 import QtGui
import sys

from PyQt5.QtCore import Qt, QRectF, QObject, pyqtSignal, QSize
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient
from PyQt5.QtWidgets import QGridLayout

from .utils import *

sys.path.append('ui')
from ftkCard import ftkCard
from .Button import arrangeButtons

class StyledIndicator(QWidget):
    def __init__(self, status=True, size=QSize(10, 10)):
        super().__init__()
        self.status = status  # True for pass, False for fail
        self.setFixedSize(size)  # Set a fixed size for the widget

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing for smoother edges

        # Define the rectangle for drawing the circle
        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)  # Slightly smaller to fit inside the widget

        # Create radial gradient for a more appealing look
        gradient = QRadialGradient(rect.center(), rect.width() / 2)
        if self.status:
            gradient.setColorAt(0, QColor(135, 255, 97))  # Light green at center
            gradient.setColorAt(1, QColor(0, 168, 57))  # Darker green at edge
        else:
            gradient.setColorAt(0, QColor(255, 136, 136))  # Light red at center
            gradient.setColorAt(1, QColor(221, 0, 0))  # Darker red at edge

        painter.setBrush(QBrush(gradient))

        # Draw the main circle with a slight border effect (optional)
        pen = QPen(Qt.gray, 0.5)  # Light gray border
        painter.setPen(pen)
        painter.drawEllipse(rect)

        # Apply the gradient fill
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(rect)


class HistoryCard(ftkCard):
    def __init__(self, parent=None, _window=None, data=None):
        super(HistoryCard, self).__init__(parent=parent)
        self.window = _window
        self.ui = _window.ui

        self.is_chosen = VariableMonitor()
        self.is_chosen.variable_changed.connect(self.update_state)

        # self.buttonList = ButtonManager(self.frame_43).buttons
        # self.buttonList = find_button(self.frame_43)
        self.data = data
        # print(data)
        self.frames = [self.frame_3,self.frame_5,self.frame_4,self.frame_6]
        self.buttons = [StyledIndicator(status=self.data['sample'][i]==1) for i in range(144)]
        # print(self.buttons)
        for frame in self.frames:
            sub_layout = QGridLayout(frame)
            sub_layout.setContentsMargins(2, 2, 2, 2)
            sub_layout.setSpacing(1)
            frame.setLayout(sub_layout)
        arrangeButtons(self.frames,self.buttons)

        self.time.setText(data['date'])
        self.label.setText(data['time'])
        self.label_2.setText(data['tip'])
        self.label_3.setText(data['tip2'])

        self.choose_sheet = '''
                            #frame{
                                border:1px solid gray;
                                border-radius:15px;
                                background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,
                                                                       stop:0 #f0f0f0, stop:0.3 #d4d4d4, stop:0.7 #b8b8b8, stop:1 #9c9c9c);
                                 border: 2px solid #007BFF;
                                 box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            }

                            #frame::hover{
                                background-color: rgb(212, 212, 212);
                            }
                        '''
        self.origin_sheet = '''
                        #frame{
                                border:1px solid gray;
                                border-radius:15px;
                            background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,
                                                                       stop:0 #f0f0f0, stop:0.3 #d4d4d4, stop:0.7 #b8b8b8, stop:1 #9c9c9c);
                                 border: 1px solid #cccccc;
                                 box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            }

                            #frame::hover{
                                background-color: rgb(212, 212, 212);
                            }
                        '''

    def update_state(self,value):
        if value:
            self.frame.setStyleSheet(self.choose_sheet)
        else:
            self.frame.setStyleSheet(self.origin_sheet)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:


        list = self.parent().findChildren(HistoryCard)
        if list:
            for item in list:
                item.is_chosen.variable = False
        self.is_chosen.variable = not self.is_chosen.variable
        # #
        # self.frame.setStyleSheet(choose_sheet)
        self.window.go_to_page(3)
        self.ui.label_26.setText('盘编号：'+self.label_2.text().split('，')[0].split('：')[1])
        self.ui.label_3.setText(self.time.text())
        self.ui.label_4.setText(self.label.text())

        status = ['pass' if self.data['sample'][i]==1 else 'fail' for i in range(len(self.data['sample']))]
        self.window.history_button.changeAllButtonType(status)



        #
        # # print(self.data)
        # for i in range(144):
        #     button = self.ui.findChild(QPushButton, f'history_{i+1}')
        #     changeSampleState(button, self.data[i])


        # print()
        # self.ui.plotBtn_3.clicked.connect(partial(self.window.plotBtnClick, self.idtray))
        # self.ui.label_3.setText(self._date)
        # self.ui.label_4.setText(self._time)
        # self.ui.label_26.setText('盘编号：' + str(self.idtray))
        # self.window.flawReInit(self.ui.frame_39)
        # self.window.history_errors = []
        # self.showHistory(self.ui.frame_39)

# class HistoryCards(QWidget):
#     def __init__(self,parent=None):
#         super().__init__(parent=parent)
#         main_frame = QFrame(self)
#         main_layout = QL(main_frame)
#         show_frame = QFrame(main_layout)
#         info_frame = QFrame(main_layout)
