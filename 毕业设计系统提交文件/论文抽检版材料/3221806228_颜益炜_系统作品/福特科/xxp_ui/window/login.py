import os
import sys
import threading
import time
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QPoint, Qt, QProcess
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsDropShadowEffect
import warnings

from .ui import MyWindow

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

processes = []

admin_user_name = 'admin'
admin_password = '123456'

def clearFile(filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write('')
        print('已清空',filepath)


class Login(QMainWindow):
    admin_user_name = admin_user_name
    admin_password = admin_password

    def __init__(self,ui_debug_queue, debug_ui_queue,ui_image_queue,image_ui_queue,exit_event):
        super().__init__()
        self.ui_debug_queue = ui_debug_queue
        self.debug_ui_queue = debug_ui_queue
        self.ui_image_queue = ui_image_queue
        self.image_ui_queue = image_ui_queue
        self.exit_event = exit_event
        self.InitUI()
        # clearFile('msg/shared_data_win_to_ZYC.txt')
        # clearFile('msg/shared_data_ZYC_to_win.txt')

    def mouseMoveEvent(self, e: QMouseEvent):  # 重写移动事件，控制窗口移动
        if self._startPos:
            self._endPos = e.pos() - self._startPos
            self.ui.move(self.ui.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        self._isTracking = True
        self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._isTracking = False
        self._startPos = None
        self._endPos = None

    def closeEvent(self, event):
        """重写关闭事件，设置退出标志"""
        self.exit_event.set()
        event.accept()

    def InitUI(self):
        self.ui: QWidget = uic.loadUi("./ui/login.ui", self)
        self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.ui.pushButton.clicked.connect(self.AdminLogin)
        self.ui.pushButton_2.clicked.connect(self.UserLogin)
        self.ui.pushButton_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))

        self.ui.lineEdit_2.returnPressed.connect(self.AdminLogin)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)  # 阴影的模糊半径
        shadow.setColor(Qt.gray)  # 阴影的颜色
        shadow.setOffset(5, 5)  # 阴影的偏移量

        self.ui.frame.setGraphicsEffect(shadow)

        self.ui.label_3.hide()

        self.ui.show()

    def AdminLogin(self):
        print('AdminLogin')
        admin = self.ui.lineEdit.text()
        password = self.ui.lineEdit_2.text()
        if admin == self.admin_user_name and password == self.admin_password:
            self.openWindow('admin')
            # process = QProcess(self)
            # process.start(r'D:\postgraduatefile\ZDevelp\Remote_Develop\ARemoteDevelop.exe')
            # # process.start(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe')
            # thread = threading.Thread(
            #     target=lambda: os.system(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe')
            # )
            # thread.start()
            #
            # window = MyWindow(mode='admin')
            # self.ui.close()
        else:
            self.ui.label_3.show()
            thread = threading.Thread(target=lambda: {time.sleep(1), self.ui.label_3.hide()})
            thread.start()

    def UserLogin(self):
        print('UserLogin')
        self.openWindow('user')
        # process = QProcess(self)
        # process.start(r'D:\postgraduatefile\ZDevelp\Remote_Develop\ARemoteDevelop.exe')
        # # process.start(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe')
        # thread = threading.Thread(
        #     target=lambda: os.system(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe'))
        # thread.start()
        # window = MyWindow(mode='user')
        # self.ui.close()
        #

    def openWindow(self, mode):
        global processes

        # process = QProcess(self)
        # process.start(r'D:\postgraduatefile\ZDevelp\Remote_Develop\ARemoteDevelop.exe')
        # processes.append(process)
        # process2 = QProcess(self)
        # process2.start(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe')
        # processes.append(process2)
        # thread = threading.Thread(
        #     target=lambda: process.start(r'D:\postgraduatefile\ZDevelp_Confocal\ZDevelp_Confocal\ZDevelop.exe'))
        # thread.start()
        # print(processes)
        self.main_window = MyWindow(self.ui_debug_queue, self.debug_ui_queue,self.ui_image_queue,self.image_ui_queue,self.exit_event,mode=mode)
        self.ui.hide()

