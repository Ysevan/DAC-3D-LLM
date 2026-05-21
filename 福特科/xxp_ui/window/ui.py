# import halcon as ha
import queue
import threading
import time
import random
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from functools import partial

import cv2
from PyQt5 import uic
from PyQt5.QtCore import QPoint, QTimer, QRect, pyqtSlot, QThread
from PyQt5.QtGui import QMouseEvent, QIcon
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QLabel, QMainWindow, QGridLayout, QPushButton, QFrame, QVBoxLayout
from PyQt5 import QtCore
from datebase.DB import DBController

from .utils import *
from .HistoryCard import HistoryCard
from collections import deque
from .Panel import SlidingPanel
from .NotificationManager import notification_manager
from .PictureViewer import ImageViewer


class MyWindow(QMainWindow):
    time1 = QTimer()  # 定时器500ms初始化
    myDatabase = DBController()  # 数据库控制类初始化
    # tcp_client = TCPClient('localhost',port=27015)
    # g = gclib.py()
    # cam = CameraFactory()
    # Zmc = ZAUXDLL()
    # light = lightControllor()

    image_queue = queue.Queue(maxsize=144)

    samples_result = []#接收样品信息的列表
    # samples = []
    sample_res = []
    stopFlag = False
    Tray_id = None

    def __init__(self, ui_debug_queue, debug_ui_queue,ui_image_queue,image_ui_queue,exit_event,mode):
        super().__init__()

        self.scan_start_time = 0
        self.history_button = None
        self.mode = mode
        self.detect_button = None
        self._startPos = None
        self._endPos = None
        self.assistant_web_process = None
        self.assistant_status_file = (
            Path(__file__).resolve().parents[3]
            / 'dac3d_iim_assistant'
            / '.tmp'
            / 'dac3d_runtime_status.json'
        )
        self.assistant_command_file = self.assistant_status_file.with_name('dac3d_assistant_command.json')
        self.assistant_command_ack_file = self.assistant_status_file.with_name('dac3d_assistant_command_ack.json')
        self._last_assistant_command_id = None
        self.assistant_latest_result = None
        self.assistant_result_history = []

        self.ui_debug_queue = ui_debug_queue
        self.debug_ui_queue = debug_ui_queue
        self.ui_image_queue = ui_image_queue
        self.image_ui_queue = image_ui_queue
        self.exit_event = exit_event

        self.InitUI(mode)
        self.writeAssistantRuntimeStatus('idle', 0, 'DAC-3D 主系统已启动，当前没有正在执行的检测任务。', 'ready')
        # self.detect_button.changeButtonType(50, 'pass')
        # self.detect_button.changeButtonType(1, 'pass')

        # print('gclib version:', self.g.GVersion())
        # self.g.GOpen('10.0.0.100 --direct -s ALL')
        #
        # strtemp = '192.168.0.11'
        # iresult = self.Zmc.ZAux_OpenEth(strtemp)
        # if 0 != iresult:
        #     print('运动控制卡连接失败')
        # else:
        #     print('运动控制卡连接成功')

    def mouseMoveEvent(self, e: QMouseEvent):  # 重写移动事件，控制窗口移动
        if self._startPos:
            self._endPos = e.pos() - self._startPos
            self.ui.move(self.ui.pos() + self._endPos)
            if not self.ui.panel.isHidden():
                self.ui.panel.move(QRect(
                    self.ui.x() + self.ui.width(),
                    self.ui.y(),
                    self.ui.panel.width(),
                    self.ui.height()
                ).topLeft())

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

    def stateChanged(self):  # 定时器循环函数
        self._time, self._date, _ = getDateAndTime()
        self.ui.label_5.setText(self._time)

        # msg = self.image_ui_queue.get(0.01)
        # if msg:
        #     pos = msg['pos']
        #     res = msg['res']
        #     if res:
        #         self.detect_button.changeButtonType(pos, 'pass')
        #     else:
        #         self.detect_button.changeButtonType(pos, 'fail')

    def InitUI(self, mode):
        # 窗口初始化

        self.ui: QWidget = uic.loadUi("./ui/ftk.ui", self)
        self.config_ui: QWidget = uic.loadUi("./ui/config.ui")
        # self.debug_ui = Debug_UI(self.g, self.cam, self.Zmc, self.light,self.image_queue)
        # self.debug_ui.hide()




        self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.history_layout = QGridLayout(self.ui.scrollAreaWidgetContents_2)
        self.ui.stackedWidget_3.hide()
        self.ui.show()


        # 消息通知管理器初始化label对象
        notification_manager.set_notification_label(self.ui.info)

        # 连接服务器
        # self.tcp_client = TCPClient('localhost', port=27015,func=self.receiveMessageFromZYC)

        # 定时器初始化，动态显示时间
        self.time1.timeout.connect(self.stateChanged)
        self.time1.start(500)
        self.assistant_command_timer = QTimer(self)
        self.assistant_command_timer.timeout.connect(self.pollAssistantCommandFile)
        self.assistant_command_timer.start(1000)

        # 所有楔形片按钮初始化
        self.buttonInit()

        # 槽函数连接及个别按钮失效初始化
        self.ui.backBtn_2.clicked.connect(self.go_back)
        self.ui.backBtn_3.clicked.connect(self.go_back)
        self.ui.backBtn_4.clicked.connect(self.go_back)
        self.ui.ConfigBtn.clicked.connect(self.ShowStatistic)
        self.ui.ConfigBtn_2.clicked.connect(self.ShowStatistic)
        self.ui.panel = SlidingPanel(self.ui,self.myDatabase)
        self.ui.historyBtn.clicked.connect(self.toggle_side_panel)
        self.ui.setBtn_3.clicked.connect(lambda :self.ui.stackedWidget_2.setCurrentIndex(4))
        self.ui.runBtn.clicked.connect(self.runBtnClicked)
        self.initOfflineControls()
        self.initAssistantWebButton()
        self.ui.allStopBtn.clicked.connect(self.stop)
        self.ui.LoginBtn.clicked.connect(self.AdminLogin)
        self.ui.label_29.hide()
        self.ui.UnloginBtn.clicked.connect(self.AdminUnlogin)
        self.ui.ConfigBtn.setEnabled(False)

        self.ui.setBtn_4.clicked.connect(self.initConfigUI)

        # 页面控制队列初始化
        self.page_history = deque([0])

        # 统计图绘制初始化
        self.figBoxXXP = FigBox()  # 单个片统计
        self.figBoxTray = FigBox()  # 全盘统计左图
        self.figBoxTrayDefection = FigBox()  # 全盘统计右图

        layout = QVBoxLayout(self.ui.frame_46)

        # 创建 ImageViewerWidget 并指定 QFrame 作为父部件
        # 使用图像路径加载图像
        self.viewer = ImageViewer(self.ui.frame_46)

        layout1 = QVBoxLayout(self.ui.frame_47)

        # 创建 ImageViewerWidget 并指定 QFrame 作为父部件
        # 使用图像路径加载图像
        self.viewer1 = ImageViewer(self.ui.frame_47)
        # self.viewer.set_image('60+15.bmp')
        # 或者使用 NumPy 数组加载图像
        # img_np = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        # self.viewer = ImageViewerWidget(img_np, self.frame)

        # 将 ImageViewerWidget 添加到 QFrame 的布局中
        layout.addWidget(self.viewer)
        layout1.addWidget(self.viewer1)

        # 设置 QFrame 的布局
        self.ui.frame_46.setLayout(layout)
        self.ui.frame_47.setLayout(layout1)
        # msg = {
        #     'func': 'run',
        #     'tray_id':self.Tray_id
        # }
        # self.tcp_client.send_message(msg)

        # 判断登录身份
        if mode == 'user':
            self.ui.setBtn.setVisible(False)
            self.ui.label_10.setVisible(False)
            self.setOfflineControlsVisible(False)
            self.ui.stackedWidget_4.setCurrentIndex(1)
        else:
            self.setOfflineControlsVisible(True)
            self.ui.stackedWidget_4.setCurrentIndex(0)
        self.ui.setBtn.clicked.connect(lambda: self.ui_debug_queue.put({
            'type': 'Info',
            'data': 'show'
        }))

        # image_process_thread = threading.Thread(target=self.getInfoFromImageProcessor)
        # image_process_thread.start()
        #
        # debug_thread = threading.Thread(target=self.getInfoFromDebug)
        # debug_thread.start()

        self.getDebug_thread = GetInfo(self.debug_ui_queue)
        self.getDebug_thread.message_received.connect(self.getInfoFromDebug)
        self.getDebug_thread.start()

        self.getImage_thread = GetInfo(self.image_ui_queue)
        self.getImage_thread.message_received.connect(self.getInfoFromImageProcessor)
        self.getImage_thread.start()



    def writeAssistantRuntimeStatus(self, state, progress=0, message='', step=None, extra=None):
        """Expose DAC-3D runtime status for the web assistant."""
        try:
            self.assistant_status_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'status': {
                    'state': state,
                    'progress': int(progress),
                    'message': message,
                    'step': step,
                    'source': 'xxp_ui',
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'mode': self.mode,
                    'tray_id': getattr(self, 'Tray_id', None),
                    'offline': bool(getattr(self, 'offlineCheckBox', None) and self.offlineCheckBox.isChecked()),
                    'offline_source_dir': getattr(self, 'offline_source_dir', ''),
                }
            }
            if self.assistant_latest_result:
                payload['status']['latest_result'] = self.assistant_latest_result
            if self.assistant_result_history:
                payload['status']['result_history'] = self.assistant_result_history
            if extra:
                payload['status'].update(extra)
            tmp_path = self.assistant_status_file.with_suffix('.tmp')
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp_path.replace(self.assistant_status_file)
        except Exception as exc:
            print(f'Write assistant runtime status failed: {exc}')

    def writeAssistantCommandAck(self, command_id, status, message, command=None):
        """Expose command handling acknowledgement for the web assistant."""
        try:
            self.assistant_command_ack_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'id': command_id,
                'status': status,
                'message': message,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'xxp_ui',
            }
            if command is not None:
                payload['command'] = command
            tmp_path = self.assistant_command_ack_file.with_suffix('.tmp')
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp_path.replace(self.assistant_command_ack_file)
        except Exception as exc:
            print(f'Write assistant command ack failed: {exc}')

    def buildAssistantLatestResult(self, sample_result, defects_num):
        """Build a JSON-safe latest result snapshot for the web assistant."""
        defections = sample_result.get('defections') or []
        normalized_defects = []
        for index, defect in enumerate(defections, start=1):
            position = defect.get('defection_pos')
            if isinstance(position, tuple):
                position = list(position)
            normalized_defects.append({
                'id': index,
                'defect_type': defect.get('defection_type', 'unknown'),
                'position': position,
                'size': defect.get('defection_size'),
                'reason': defect.get('reason', ''),
            })

        pos = sample_result.get('pos')
        quality = bool(sample_result.get('quality'))
        result_pic_path = sample_result.get('result_pic_path') or {}
        first_defect = normalized_defects[0] if normalized_defects else None
        parsed_result = {
            'defect_type': first_defect.get('defect_type', 'none') if first_defect else 'none',
            'location': f"pos{pos}" if pos is not None else 'latest_sample',
            'confidence': 1.0,
            'measurements': {},
            'severity': 'none' if quality else 'medium',
            'rule_reason': '最近样品未发现不合格缺陷。' if quality else (
                first_defect.get('reason') or f'最近样品发现 {defects_num} 个不合格缺陷。'
            ),
        }
        if first_defect and isinstance(first_defect.get('size'), (int, float)):
            parsed_result['measurements']['size_px'] = float(first_defect['size'])

        return {
            'result_root': str(Path(str(result_pic_path.get('surface_1', ''))).parent) if result_pic_path else '',
            'tray_id': self.Tray_id,
            'position': pos,
            'quality': quality,
            'quality_label': '合格' if quality else '不合格',
            'defects_num': defects_num,
            'defects': normalized_defects,
            'files': [path for path in result_pic_path.values() if path],
            'result_pic_path': result_pic_path,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parsed_result': parsed_result,
        }

    def pollAssistantCommandFile(self):
        """Poll assistant-generated command files and reflect them in DAC-3D UI state."""
        try:
            if not self.assistant_command_file.exists():
                return
            payload = json.loads(self.assistant_command_file.read_text(encoding='utf-8-sig'))
            command_id = payload.get('id')
            if not command_id or command_id == self._last_assistant_command_id:
                return
            command = payload.get('command') or {}
            if not isinstance(command, dict):
                return

            self._last_assistant_command_id = command_id
            action = str(command.get('action') or '')
            message = self.describeAssistantCommand(command)
            notification_manager.show_notification(message)

            if action == 'start_online_scan':
                if not self.ui.runBtn.isEnabled():
                    busy_message = '当前已有检测任务正在运行，拒绝重复启动在线扫描。'
                    notification_manager.show_notification(busy_message)
                    self.writeAssistantCommandAck(command_id, 'rejected', busy_message, command)
                    self.writeAssistantRuntimeStatus('command_rejected', 0, busy_message, 'assistant_command_rejected', {'assistant_command': command})
                    return
                self.writeAssistantCommandAck(command_id, 'accepted', '已接收在线扫描命令，正在启动在线检测。', command)
                self.writeAssistantRuntimeStatus('command_received', 0, message, 'assistant_command_received', {'assistant_command': command})
                if self.offlineCheckBox.isChecked():
                    self.offlineCheckBox.setChecked(False)
                self.runBtnClicked()
                return

            if action == 'start_offline_detection':
                if not self.ui.runBtn.isEnabled():
                    busy_message = '当前已有检测任务正在运行，拒绝重复启动离线检测。'
                    notification_manager.show_notification(busy_message)
                    self.writeAssistantCommandAck(command_id, 'rejected', busy_message, command)
                    self.writeAssistantRuntimeStatus('command_rejected', 0, busy_message, 'assistant_command_rejected', {'assistant_command': command})
                    return
                image_folder = ((command.get('payload') or {}).get('image_folder') or '').strip()
                if not image_folder:
                    fail_message = '助手命令缺少离线图片目录，无法启动离线检测。'
                    notification_manager.show_notification(fail_message)
                    self.writeAssistantCommandAck(command_id, 'rejected', fail_message, command)
                    self.writeAssistantRuntimeStatus('command_rejected', 0, fail_message, 'assistant_command_rejected', {'assistant_command': command})
                    return
                self.offline_source_dir = image_folder
                self.offlineCheckBox.setChecked(True)
                self.writeAssistantCommandAck(command_id, 'accepted', f'已接收离线检测命令：{image_folder}', command)
                self.writeAssistantRuntimeStatus('command_received', 0, message, 'assistant_command_received', {'assistant_command': command})
                self.runBtnClicked()
                return

            if action == 'stop_detection':
                self.writeAssistantCommandAck(command_id, 'accepted', '已接收停止检测命令。', command)
                self.stop()
                return

            self.writeAssistantCommandAck(command_id, 'previewed', message, command)
            self.writeAssistantRuntimeStatus(
                'command_received',
                0,
                message,
                'assistant_command_received',
                {'assistant_command': command},
            )
        except Exception as exc:
            print(f'Poll assistant command failed: {exc}')

    def describeAssistantCommand(self, command):
        """Build a concise operator-facing description for an assistant command."""
        action = str(command.get('action') or 'unknown')
        if action == 'scan':
            area = command.get('scan_area_mm') or {}
            width = area.get('width', '未知')
            height = area.get('height', '未知')
            mode = command.get('mode') or 'standard'
            region = command.get('region') or 'current_selection'
            return f'智能助手收到扫描命令预览：{width}mm × {height}mm，区域 {region}，模式 {mode}。请在主系统确认后执行。'
        if action == 'start_online_scan':
            return '智能助手请求启动在线 144 点扫描。'
        if action == 'start_offline_detection':
            image_folder = (command.get('payload') or {}).get('image_folder') or '未指定目录'
            return f'智能助手请求启动离线检测：{image_folder}'
        if action == 'stop_detection':
            return '智能助手请求停止当前检测。'
        if action == 'query_status':
            return '智能助手请求查询当前检测状态。'
        if action == 'get_latest_result':
            return '智能助手请求读取最近一次检测结果。'
        if action == 'validate_offline_folder':
            image_folder = (command.get('payload') or {}).get('image_folder') or '未指定目录'
            return f'智能助手请求校验离线图片目录：{image_folder}'
        return f'智能助手收到结构化命令：{action}'


    def initAssistantWebButton(self):
        """Add a stable web entry for the LLM assistant."""
        self.assistant_web_url = 'http://127.0.0.1:7860'
        admin_page = self.ui.stackedWidget_4.widget(0)

        self.assistantToolbarFrame = QFrame(self.ui.frame_4)
        self.assistantToolbarFrame.setObjectName('assistantToolbarFrame')
        toolbar_layout = QVBoxLayout(self.assistantToolbarFrame)
        toolbar_layout.setContentsMargins(6, 0, 6, 0)
        toolbar_layout.setSpacing(0)

        self.assistantWebBtn = QPushButton('AI', self.assistantToolbarFrame)
        self.assistantWebBtn.setObjectName('assistantWebBtn')
        self.assistantWebBtn.setFixedSize(66, 58)

        assistant_button_style = """
            QPushButton {
                background-color: rgb(219, 245, 255);
                color: rgb(18, 85, 145);
                border: 1px solid rgb(120, 202, 235);
                border-radius: 14px;
                font: 18pt "Arial";
                font-weight: 800;
            }
            QPushButton:hover {
                background-color: rgb(190, 235, 255);
                border: 1px solid rgb(58, 165, 222);
            }
            QPushButton:pressed {
                background-color: rgb(165, 222, 250);
            }
        """
        self.assistantWebBtn.setStyleSheet(assistant_button_style)
        self.assistantWebBtn.clicked.connect(self.openAssistantWeb)

        toolbar_layout.addWidget(self.assistantWebBtn, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.ui.horizontalLayout_7.addWidget(self.assistantToolbarFrame)

        self.assistantAdminBtn = QPushButton('智能助手', admin_page)
        self.assistantAdminBtn.setObjectName('assistantAdminBtn')
        self.assistantAdminBtn.setGeometry(380, 292, 150, 40)
        self.assistantAdminBtn.setStyleSheet(
            'QPushButton{border:1px solid rgb(135, 203, 232); border-radius:8px; '
            'font: 15pt "楷体"; font-weight:700; background-color: rgb(238, 250, 255); '
            'color: rgb(20, 74, 122);}'
            'QPushButton:hover{background-color: rgb(208, 241, 255); border-color: rgb(75, 171, 220);}'
            'QPushButton:pressed{background-color: rgb(186, 229, 249);}'
        )
        self.assistantAdminBtn.clicked.connect(self.openAssistantWeb)
        self.assistantAdminBtn.raise_()
        self.assistantAdminBtn.show()

    def openAssistantWeb(self):
        """Start both assistant web UIs and open the selected target."""
        url = getattr(self, 'assistant_web_url', 'http://127.0.0.1:7860')
        try:
            assistant_root = Path(__file__).resolve().parents[3] / 'dac3d_iim_assistant'
            app_py = assistant_root / 'app.py'
            if app_py.exists() and (
                self.assistant_web_process is None or self.assistant_web_process.poll() is not None
            ):
                creationflags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0)
                env = os.environ.copy()
                env['DAC3D_ENDPOINT'] = self.assistant_status_file.resolve().as_uri()
                env['DAC3D_COMMAND_PATH'] = str(self.assistant_command_file.resolve())
                self.assistant_web_process = subprocess.Popen(
                    [sys.executable, str(app_py)],
                    cwd=str(assistant_root),
                    env=env,
                    creationflags=creationflags,
                )
                time.sleep(2.0)
            webbrowser.open(url)
            notification_manager.show_notification(f'已打开智能助手网页：{url}')
        except Exception as exc:
            print(f'Open assistant web failed: {exc}')
            notification_manager.show_notification(f'智能助手网页启动失败: {exc}')

    def initOfflineControls(self):
        admin_page = self.ui.stackedWidget_4.widget(0)
        self.offline_source_dir = ''

        self.offlineCheckBox = QCheckBox('离线检测', admin_page)
        self.offlineCheckBox.setObjectName('offlineCheckBox')
        self.offlineCheckBox.setGeometry(80, 295, 120, 34)
        self.offlineCheckBox.setStyleSheet(
            'QCheckBox{font: 15pt "楷体"; color: black;}'
            'QCheckBox::indicator{width:18px; height:18px;}'
        )

        self.chooseOfflineDirBtn = QPushButton('选择原图目录', admin_page)
        self.chooseOfflineDirBtn.setObjectName('chooseOfflineDirBtn')
        self.chooseOfflineDirBtn.setGeometry(210, 292, 150, 40)
        self.chooseOfflineDirBtn.setStyleSheet(
            'QPushButton{border:none; border-radius:8px; font: 15pt "楷体"; '
            'background-color: rgb(208, 251, 255); color: black;}'
            'QPushButton:hover{background-color: rgb(188, 231, 235);}'
        )
        self.chooseOfflineDirBtn.clicked.connect(self.chooseOfflineSourceDir)

        self.offlineDirLabel = QLabel('未选择目录', admin_page)
        self.offlineDirLabel.setObjectName('offlineDirLabel')
        self.offlineDirLabel.setGeometry(80, 352, 320, 36)
        self.offlineDirLabel.setStyleSheet('font: 13pt "楷体"; color: rgb(60, 60, 60);')
        self.offlineDirLabel.setWordWrap(True)

    def setOfflineControlsVisible(self, visible):
        for widget in (
            getattr(self, 'offlineCheckBox', None),
            getattr(self, 'chooseOfflineDirBtn', None),
            getattr(self, 'assistantAdminBtn', None),
            getattr(self, 'offlineDirLabel', None),
        ):
            if widget is not None:
                widget.setVisible(visible)

    def chooseOfflineSourceDir(self):
        folder = QFileDialog.getExistingDirectory(self.ui, '选择离线相机原图文件夹', '')
        if not folder:
            return
        self.offline_source_dir = folder
        self.offlineCheckBox.setChecked(True)
        display_folder = folder
        if len(display_folder) > 28:
            display_folder = '...' + display_folder[-28:]
        self.offlineDirLabel.setText(display_folder)
        notification_manager.show_notification('已选择离线原图目录')



        # panel.toggle_side_panel()

    def toggle_side_panel(self):
        if self.ui.panel.isHidden():
            # if self.Tray_id:
            #     self.ui.panel.addNewTrayHistory(self.Tray_id)
            self.ui.panel.show_animation(self.ui.geometry())
        else:
            self.ui.panel.hide_animation(self.ui.geometry())
            self.go_to_page(0)
            list = self.ui.panel.findChildren(HistoryCard)
            if list:
                for item in list:
                    item.is_chosen.variable = False

    def initConfigUI(self):
        print('config')
        buttonList = find_button(self.config_ui.frame_30)
        sheet = '''
                QPushButton{
                    border:1px solid black;
                    border-radius:15px;
                    background-color: rgb(217, 217, 217);

                    font: 10pt "黑体";
                }
                QPushButton:hover {
                    background-color: #555;  /* 鼠标悬停时的背景颜色 */
                    border-style: inset;
                }

                QPushButton:pressed {
                    background-color: #777;  /* 鼠标点击时的背景颜色 */
                    border-style: inset;
                }
                '''
        for (i, item) in enumerate(buttonList):
            item.setIcon(QIcon(None))
            item.setObjectName(f'config_{i + 1}')
            item.setText(f'{i + 1}')
            # print(f'func:positioning,number:{i + 1}')
            item.clicked.connect(partial(self.configSendMsg, i + 1))
            item.setStyleSheet(sheet)
            # item.setEnabled(False)
        self.config_ui.show()

    def configSendMsg(self, num):
        msg = {
            'type' : 'Control',
            'func': 'Move',
            'button_num': num
        }
        self.ui_debug_queue.put(msg)
        # self.tcp_client.send_message(msg)
        # sendDataToZYC(f'func:positioning,number:{num}')f

    def buttonInit(self):
        self.detect_button = initButtons(self.ui,self.ui.frame_30,30,'detect')
        # detect_button.changeButtonType(30,'pass')

        self.history_button = initButtons(self.ui,self.ui.frame_38,30,'history')

        # main_frame = QFrame(self)
        # main_layout = QVBoxLayout()
        # main_layout.addWidget(main_frame)
        # Button = Buttons(main_frame, 30)
        # self.ui.frame_30.setLayout(main_layout)
        # detectButtonManager = ButtonManager(self.ui.frame_30)
        # buttonList = detectButtonManager.buttons
        # for (i, item) in enumerate(buttonList):
        #     item.setIcon(QIcon(None))
        #     item.setObjectName(f'detect_{i + 1}')
        #     item.clicked.connect(self.XXPResShowBtnClick)
        #     item.setEnabled(False)
        # historyButtonManager = ButtonManager(self.ui.frame_38)
        # buttonList = historyButtonManager.buttons
        # for (i, item) in enumerate(buttonList):
        #     item.setIcon(QIcon(None))
        #     item.setObjectName(f'detect_{i + 1}')
        #     item.clicked.connect(self.XXPResShowBtnClick)

        # buttonList = find_button(self.ui.frame_30)
        # sheet = '''
        # QPushButton{
        #     border:1px solid black;
        #     border-radius:15px;
        #     background-color: rgb(217, 217, 217);
        #
        #     font: 10pt "黑体";
        # }
        # '''
        # for (i, item) in enumerate(buttonList):
        #     item.setIcon(QIcon(None))
        #     item.setObjectName(f'detect_{i + 1}')
        #     item.setText(f'{i + 1}')
        #     item.clicked.connect(self.XXPResShowBtnClick)
        #     item.setStyleSheet(sheet)
        #     item.setEnabled(False)
        #
        # buttonList = find_button(self.ui.frame_38)
        # for (i, item) in enumerate(buttonList):
        #     item.setIcon(QIcon(None))
        #     item.setObjectName(f'history_{i + 1}')
        #     item.setText(f'{i + 1}')
        #     item.clicked.connect(self.XXPResShowBtnClick)
        #     item.setStyleSheet(sheet)

        # --test
        # button = self.findChild(QPushButton, 'detect_1')
        # changeSampleState(button, 1)

    @pyqtSlot(dict)
    def getInfoFromImageProcessor(self,msg): # 获取从图像处理进程发来的消息
        print(msg)
        if msg['type'] == 'Sample_detection_result':

            sample_result = msg['sample_data']

            pos = sample_result['pos']
            res = sample_result['quality']
            result_pic_path = sample_result['result_pic_path']
            result_img = sample_result['result_img']
            defections = sample_result['defections']
            defects_num = len(defections)
            self.assistant_latest_result = self.buildAssistantLatestResult(sample_result, defects_num)
            self.assistant_result_history.append(self.assistant_latest_result)
            self.assistant_result_history = self.assistant_result_history[-144:]


            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",pos,res,self.Tray_id,result_pic_path)
            data = {'Tray_id': self.Tray_id, 'quality': res, 'pic_path': result_pic_path['surface_1']+'*'+result_pic_path['surface_2'], 'pos': pos, 'defection_num': defects_num}
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",data)
            self.myDatabase.addNewSampleToDB(data)
            sample_id = self.myDatabase.getLastSampleId() - 1
            thread = threading.Thread(target=self.saveDefectionsToDB,args=(sample_id,defections,))
            thread.start()

            sample={
                'res':res,
                'result_pic_path':result_pic_path,
                'result_img':result_img
            }

            self.samples_result.append(sample)

            if res:
                self.detect_button.changeButtonType(pos, 'pass')
            else:
                self.detect_button.changeButtonType(pos, 'fail')
            self.writeAssistantRuntimeStatus(
                'running',
                round(pos / 144 * 100),
                f'检测中...({pos}/144)，当前样品判定: {"合格" if res else "不合格"}，缺陷数量: {defects_num}',
                'sample_detection_result',
                {
                    'current_position': pos,
                    'total_positions': 144,
                    'quality': bool(res),
                    'defects_num': defects_num,
                    'result_pic_path': result_pic_path,
                    'latest_result': self.assistant_latest_result,
                    'result_history': self.assistant_result_history,
                },
            )
            notification_manager.show_notification(f'检测中...({pos}/144)')
        if msg['type']=='info':
            if msg['data']=='finish':
                end_time = time.time()
                self.writeAssistantRuntimeStatus(
                    'completed',
                    100,
                    f'检测完成，检测用时：{end_time-self.scan_start_time:.2f}秒',
                    'finished',
                    {
                        'elapsed_seconds': round(end_time - self.scan_start_time, 2),
                        'latest_result': self.assistant_latest_result,
                        'result_history': self.assistant_result_history,
                    },
                )
                notification_manager.show_notification(f'检测完成，检测用时：{end_time-self.scan_start_time}秒')
                self.ui.ConfigBtn.setEnabled(True)
                self.ui.runBtn.setEnabled(True)
                self.chooseOfflineDirBtn.setEnabled(True)
                self.offlineCheckBox.setEnabled(True)
        # while True:
        #     try:
        #         msg = self.image_ui_queue.get(2)
        #         if msg:
        #             print(msg)
        #             if msg['type']=='Sample_detection_result':
        #
        #                 sample_result = msg['sample_data']
        #
        #                 pos = sample_result['pos']
        #                 res = sample_result['quality']
        #                 result_pic_path = sample_result['result_pic_path']
        #
        #                 sample={
        #                     'res':res,
        #                     'result_pic_path':result_pic_path
        #                 }
        #
        #                 self.samples_result.append(sample)
        #
        #                 if res:
        #                     self.detect_button.changeButtonType(pos, 'pass')
        #                 else:
        #                     self.detect_button.changeButtonType(pos, 'fail')
        #                 notification_manager.show_notification(f'检测中...({pos}/144)')
        #             if msg['type']=='info':
        #                 notification_manager.show_notification(f'检测完成，检测用时：')
        #
        #
        #     except queue.Empty:
        #         pass

    def saveDefectionsToDB(self,sample_id,defections):
        for defection in defections:
            data = {
                'Sample_id': sample_id,
                'defection_type': defection['defection_type'],
                'defection_pos': str(defection['defection_pos']),
                'defection_size': defection["defection_size"]}
            self.myDatabase.addNewDefectionToDB(data)

    @pyqtSlot(dict)
    def getInfoFromDebug(self,msg): # 获取从debug发来的消息
        data = msg['data']
        if msg['type'] == 'info':
            self.scan_start_time = data['start_time']
        if msg['type'] == 'hardware_info':
            notification_manager.show_notification('设备初始化完成！')
        # try:
        #     msg = self.debug_ui_queue.get(2)
        #     if msg:
        #         if msg['type'] == 'info':
        #             data = msg['data']
        #             self.scan_start_time = data['start_time']
        #             print(f'检测开始时间为：{self.scan_start_time}')
        #
        # except queue.Empty:
        #     pass
    # @pyqtSlot(str)
    # def receiveMessageFromZYC(self,message):
    #     if message:
    #         # try:
    #             # Convert JSON string to Python dictionary
    #             msg = json.loads(message)
    #             print('接收到服务器消息：',msg)
    #             # try:
    #             if not self.stopFlag:
    #                 if isinstance(msg, dict):
    #                     data = msg['data']
    #                     if msg['type'] == 'new_sample':
    #
    #                         # self.samples.append(data['pic_path'])
    #                         img = ha.read_image(data['pic_path'])
    #                         res,quality,defections = detect(img)
    #                         self.viewer.set_image(res)
    #                         if not quality:
    #                             self.detect_button.changeButtonType(data['pos'], 'fail')
    #                         else:
    #                             self.detect_button.changeButtonType(data['pos'], 'pass')
    #
    #                         sample_data = {
    #                                 'Tray_id': data['tray_id'],
    #                                 'quality': int(quality),#1--合格，0--不合格
    #                                 'pic_path': data['pic_path'],
    #                                 'pos': data['pos'],
    #                                 'defection_num': len(defections)
    #                             }
    #                         # print(sample_data)
    #                         self.myDatabase.addNewSampleToDB(sample_data)
    #                         Sample_id = self.myDatabase.getLastSampleId()
    #                         defection_data = []
    #                         if defections:
    #                             for defection in defections:
    #                                 temp = {
    #                                     'Sample_id': Sample_id,
    #                                     'defection_type': defection['defection_type'],
    #                                     'defection_pos': json.dumps(defection['defection_pos']),
    #                                     'defection_size': defection['defection_size'],
    #                                 }
    #                                 defection_data.append(temp)
    #                                 self.myDatabase.addNewDefectionToDB(temp)
    #                             notification_manager.show_notification(f'检测中...({data["pos"]}/144)')
    #                         self.samples.append({
    #                                 'sample_data':sample_data,
    #                                 'defection_data':defection_data
    #                             })
    #                         if data["pos"] >= 144:
    #                             notification_manager.show_notification('检测完成(144/144)')
    #                             self.ui.ConfigBtn.setEnabled(True)
    #                             self.ui.runBtn.setEnabled(True)
    #                             self.ui.panel.addNewTrayHistory(self.Tray_id)
    #
    #
    #                             # image_data = base64.b64decode(data['image'])
    #                             # # 将二进制数据转换为 NumPy 数组
    #                             # nparr = np.frombuffer(image_data, np.uint8)
    #                             # # 使用 OpenCV 解码图像
    #                             # image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    #                             # image_halcon = ha.himage_from_numpy_array(image)
    #                             # res = detect(image_halcon)
    #                             # cv2.imwrite('out.bmp',res)
    #
    #                         # if msg['type'] == 'new_sample':
    #                         #     sample_data = {
    #                         #         'Tray_id': data['tray_id'],
    #                         #         'quality': data['quality'],
    #                         #         'pic_path': data['pic_path'],
    #                         #         'pos': data['pos'],
    #                         #         'defection_num': len(data['defection'])
    #                         #     }
    #                         #     self.myDatabase.addNewSampleToDB(sample_data)
    #                         #     Sample_id = self.myDatabase.getLastSampleId()
    #                         #     pos = data['pos']
    #                         #     if data['quality'] == 0:
    #                         #         self.detect_button.changeButtonType(pos,'fail')
    #                         #     else:
    #                         #         self.detect_button.changeButtonType(pos,'pass')
    #                         #     defection_data = []
    #                         #     if data['defection']:
    #                         #         for defection in data['defection']:
    #                         #             temp = {
    #                         #                 'Sample_id': Sample_id,
    #                         #                 'defection_type': defection['defection_type'],
    #                         #                 'defection_pos': json.dumps(defection['defection_pos']),
    #                         #                 'defection_size': defection['defection_size'],
    #                         #             }
    #                         #             defection_data.append(temp)
    #                         #             self.myDatabase.addNewDefectionToDB(temp)
    #                         #         notification_manager.show_notification(f'检测中...({data["pos"]}/144)')
    #                         #     self.samples.append({
    #                         #         'sample_data':sample_data,
    #                         #         'defection_data':defection_data
    #                         #     })
    #                         #
    #                         #
    #                         #     if data["pos"] >= 144:
    #                         #         notification_manager.show_notification('检测完成(144/144)')
    #                         #         self.ui.ConfigBtn.setEnabled(True)
    #                         #         self.ui.runBtn.setEnabled(True)
    #                         #         self.ui.panel.addNewTrayHistory(self.Tray_id)
    #
    #         #     except Exception as e:
    #         #         print(e)
    #         #
    #         # except json.JSONDecodeError as e:
    #         #     print(f"Failed to decode JSON: {e}")
    #
    #     # self.ZYCMsg = message
    #     # if self.ZYCMsg:
    #     #     # print('接收到服务器消息：', self.ZYCMsg)
    #     #     try:
    #     #         if not self.stopFlag:
    #     #             if isinstance(self.ZYCMsg, dict):
    #     #                 data = self.ZYCMsg['data']
    #     #                 if self.ZYCMsg['type'] == 'new_sample':
    #     #                     # print(data['defection'])
    #     #                     sample_data = {
    #     #                         'Tray_id': data['tray_id'],
    #     #                         'quality': data['quality'],
    #     #                         'pic_path': data['pic_path'],
    #     #                         'pos': data['pos'],
    #     #                         'defection_num': len(data['defection'])
    #     #                     }
    #     #
    #     #                     Sample_id = self.myDatabase.getLastSampleId()
    #     #                     pos = data['pos']
    #     #                     # button = self.findChild(QPushButton, f'detect_{pos}')
    #     #                     # changeSampleState(button, int(data['quality']))
    #     #                     if data['quality'] == 0:
    #     #                         self.detect_button.changeButtonType(pos,'fail')
    #     #                     else:
    #     #                         self.detect_button.changeButtonType(pos,'pass')
    #     #                     defection_data = {}
    #     #                     if data['defection']:
    #     #                         for defection in data['defection']:
    #     #                             defection_data = {
    #     #                                 'Sample_id': Sample_id,
    #     #                                 'defection_type': defection['defection_type'],
    #     #                                 'defection_pos': json.dumps(defection['defection_pos']),
    #     #                                 'defection_size': defection['defection_size'],
    #     #                             }
    #     #                             # print(defection_data)
    #     #                             # self.myDatabase.addNewDefectionToDB(defection_data)
    #     #                         self.ui.info.setText(f'检测中...({data["pos"]}/144)')
    #     #                     self.samples.append({
    #     #                         'sample_data':sample_data,
    #     #                         'defection_data':defection_data
    #     #                     })
    #     #
    #     #
    #     #                     if data["pos"] >= 144:
    #     #                         self.ui.info.setText('检测完成(144/144)')
    #     #                         self.ui.ConfigBtn.setEnabled(True)
    #     #                         self.ui.runBtn.setEnabled(True)
    #     #                         DBThread = threading.Thread(target=self.addSampleToDB)
    #     #                         DBThread.start()
    #     #
    #     #     except Exception as e:
    #     #         print(e)


    def go_to_page(self, index):
        if index != self.page_history[-1]:
            self.page_history.append(index)
        self.ui.stackedWidget_2.setCurrentIndex(index)

    def go_back(self):
        if len(self.page_history) > 1:
            # 移除当前页面索引
            self.page_history.pop()
            # 设置前一个页面为当前页面
            previous_index = self.page_history[-1]
            self.ui.stackedWidget_2.setCurrentIndex(previous_index)
            if previous_index == 0:  # 若回退页为首页，关闭历史记录栏
                if not self.ui.panel.isHidden():
                    self.toggle_side_panel()

    def XXPResShowBtnClick(self):#数字按钮
        self.go_to_page(1)
        button = self.ui.sender()

        if button:
            buttonName = button.objectName()
            button_num = buttonName.split('_')[1]
            print(button_num)
            button_num=int(button_num)-1

            # msg = {
            #     'type':'Control',
            #     'func':'Move',
            #     'button_num':button_num-1
            # }
            # self.ui_debug_queue.put(msg)
            print(self.samples_result)
            # result_pic_path =
            self.viewer.display_image(self.samples_result[button_num]['result_img'][0]) #界面展示图像
            self.viewer1.display_image(self.samples_result[button_num]['result_img'][1])  # 界面展示图像
            # cv2.imshow(f'{button_num}',result_pic_path)
            # cv2.waitKey(0)




            #
            # self.ui.label_28.setText('样品位置编号:' + str(button_num))
            # if buttonName.split('_')[0] == 'detect':
            #     Tray_id = self.Tray_id
            # else:
            #     Tray_id = int(self.ui.label_26.text().split('：')[1])
            # #
            # self.ui.label_27.setText(f'盘编号：{Tray_id}')
            # time_info = self.myDatabase.getTrayDetectTime(Tray_id)
            # self.ui.label_30.setText(f'时间：{time_info}')
            #
            # count_splash = 0
            # count_scratch = 0
            # count_chipping = 0
            # def_info = self.myDatabase.getDefectionByPos(Tray_id, int(button_num))
            # if def_info:
            #     for defection in def_info:
            #         if defection['defection_type'] == 'splash':
            #             count_splash += 1
            #         elif defection['defection_type'] == 'scratch':
            #             count_scratch += 1
            #         else:
            #             count_chipping += 1
            #
            #     # data = {
            #     #     'x': ['麻点', '划痕', '崩边'],
            #     #     'y': [count_splash, count_scratch, count_chipping],
            #     #     'title': '单片样品瑕疵数量统计图'
            #     # }
            #     # data = {'麻点': count_splash, '划痕': count_scratch, '崩边': count_chipping}
            #     data = {
            #         'data': {'麻点': count_splash, '划痕': count_scratch, '崩边': count_chipping},
            #         'title': '单片样品瑕疵数量统计图'
            #     }
            #     plotFig(self.ui.frame_113, data, self.figBoxXXP)
            #     # plotFig(data, self.ui, self.ui.frame_113)
            # else:
            #     # data = {
            #     #     'x': ['麻点', '划痕', '崩边'],
            #     #     'y': [0, 0, 0],
            #     #     'title': '无瑕疵'
            #     # }
            #     data = {
            #         'data': {'麻点': 0, '划痕': 0, '崩边': 0},
            #         'title': '无瑕疵'
            #     }
            #     plotFig(self.ui.frame_113, data, self.figBoxXXP)

            # self.debug_ui.g.GCommand(f'pPOSX={ self.debug_ui.xy_pos[button_num][0]}')
            # self.debug_ui.g.GCommand('XQ  # MOVEX,1')
            # self.debug_ui.g.GCommand(f'pPOSY={ self.debug_ui.xy_pos[button_num][1]}')
            # self.debug_ui.g.GCommand('XQ  # MOVEY,2')

        # if button:
        #     buttonName = button.objectName()
        #     print(f'{buttonName} was clicked')
        #     button_num = buttonName.split('_')[1]
        #     if buttonName.split('_')[0] == 'detect':
        #         Tray_id = self.Tray_id
        #     else:
        #         Tray_id = int(self.ui.label_26.text().split('：')[1])
        #     time_info = self.myDatabase.getTrayDetectTime(Tray_id)
        #
        #     self.ui.label_28.setText('样品位置编号:' + button_num)
        #
        #     self.ui.label_27.setText(f'盘编号：{Tray_id}')
        #     self.ui.label_30.setText(f'时间：{time_info}')
        #     count_splash = 0
        #     count_scratch = 0
        #     count_chipping = 0
        #     def_info = self.myDatabase.getDefectionByPos(Tray_id, int(button_num))
        #     if def_info:
        #         for defection in def_info:
        #             if defection['defection_type'] == 'splash':
        #                 count_splash += 1
        #             elif defection['defection_type'] == 'scratch':
        #                 count_scratch += 1
        #             else:
        #                 count_chipping += 1
        #
        #         # data = {
        #         #     'x': ['麻点', '划痕', '崩边'],
        #         #     'y': [count_splash, count_scratch, count_chipping],
        #         #     'title': '单片样品瑕疵数量统计图'
        #         # }
        #         # data = {'麻点': count_splash, '划痕': count_scratch, '崩边': count_chipping}
        #         data = {
        #             'data': {'麻点': count_splash, '划痕': count_scratch, '崩边': count_chipping},
        #             'title': '单片样品瑕疵数量统计图'
        #         }
        #         plotFig(self.ui.frame_113, data, self.figBoxXXP)
        #         # plotFig(data, self.ui, self.ui.frame_113)
        #     else:
        #         # data = {
        #         #     'x': ['麻点', '划痕', '崩边'],
        #         #     'y': [0, 0, 0],
        #         #     'title': '无瑕疵'
        #         # }
        #         data = {
        #             'data': {'麻点': 0, '划痕': 0, '崩边': 0},
        #             'title': '无瑕疵'
        #         }
        #         plotFig(self.ui.frame_113, data, self.figBoxXXP)
        #         # y_ticks = self.figBoxXXP._dynamic_ax.get_yticks()
        #         # self.figBoxXXP._dynamic_ax.set_yticks(y_ticks)

    def ShowStatistic(self):

        if self.ui.stackedWidget_2.currentIndex() == 3:
            Tray_id = int(self.ui.label_26.text().split('：')[1])
        else:
            Tray_id = self.Tray_id

        self.go_to_page(2)
        # self.switch_page(2)
        # info = self.myDatabase.getSampleBelongToTray(Tray_id)
        time_info = self.myDatabase.getTrayDetectTime(Tray_id)
        good_count,bad_count = self.myDatabase.countSampleOfTray(Tray_id)
        defections_count =  self.myDatabase.countDefectionOfSample(Tray_id)
        self.ui.label_43.setText(f'日期：{time_info.split(" ")[0]}')
        self.ui.label_44.setText(f'时间：{time_info.split(" ")[1]}')
        self.ui.label_45.setText(f'盘编号：{Tray_id}')

        data = {
            'data': {'合格品': good_count, '不合格品': bad_count},
            'title': '整盘样品情况统计图'
        }

        plotFig(self.ui.groupBox, data, self.figBoxTray)

        count_splash = 0
        count_scratch = 0
        count_chipping = 0
        for detection in defections_count:
            if detection['defection_type'] == 'splash':
                count_splash = detection['defect_count']
            elif detection['defection_type'] == 'scratch':
                count_scratch = detection['defect_count']
            elif detection['defection_type'] == 'chipping':
                count_chipping = detection['defect_count']
        data = {
            'data': {'麻点': count_splash, '划痕': count_scratch, '崩边': count_chipping},
            'title': '整盘瑕疵种类数量统计图'
        }
        plotFig(self.ui.groupBox_2, data, self.figBoxTrayDefection)


    # def hideSetBtnClick(self):
    #     self.isHistoryShowing = False
    #     self.myDatabase.history_begin_num = 0
    #     self.ui.move(self.l + 300, self.t)
    #     self.ui.stackedWidget_3.setVisible(False)
    #     self.ui.stackedWidget_2.setCurrentIndex(0)
    #     sheet = """
    #             #frame{
    #                     border-radius:30px;
    #                     background-color: rgb(244, 244, 244);
    #                   }
    #             """
    #     self.ui.frame.setStyleSheet(sheet)
    #     # list = self.ui.scrollAreaWidgetContents_2.findChildren(HistoryCard)
    #     # if list:
    #     #     for item in list:
    #     #         item.deleteLater()


    def runBtnClicked(self):
        print('检测开始')
        if self.offlineCheckBox.isChecked():
            if not self.offline_source_dir:
                notification_manager.show_notification('请先选择离线原图目录')
                return
            self.startDetectionRun()
            self.writeAssistantRuntimeStatus(
                'running',
                0,
                f'离线检测已启动，正在读取原图目录: {self.offline_source_dir}',
                'offline_detect_folder',
                {'offline_source_dir': self.offline_source_dir},
            )
            self.ui_image_queue.put({
                'type': 'offline_detect_folder',
                'data': {
                    'folder': self.offline_source_dir,
                    'tray_id': self.Tray_id
                }
            })
            notification_manager.show_notification('离线检测中...(0/144)')
            return

        self.startDetectionRun()
        msg = {
            'type':'Control',
            'func':'Scan'
        }
        self.writeAssistantRuntimeStatus('running', 0, '在线扫描已启动，等待硬件扫描与图像采集。', 'online_scan')
        self.ui_debug_queue.put(msg)

    def offlineDetectClicked(self):
        self.chooseOfflineSourceDir()

    def startDetectionRun(self):
        self.scan_start_time = time.time()
        self.stopFlag = False
        if not self.ui.panel.isHidden():
            self.ui.panel.hide_animation(self.ui.geometry())
        self.detect_button.changeAllButtonType(['pending']*144)
        self.samples_result = []
        self.previous_index = self.stackedWidget_2.currentIndex()
        self.ui.stackedWidget_2.setCurrentIndex(0)
        notification_manager.show_notification('检测中...(0/144)')
        # self.ui.info.setText('检测中...(0/144)')
        self.ui.runBtn.setEnabled(False)
        self.chooseOfflineDirBtn.setEnabled(False)
        self.offlineCheckBox.setEnabled(False)

        self.Tray_id = self.myDatabase.getLastTrayId()
        self.ui.plateNum.setText(f'盘编号：{self.Tray_id}')
        self.ui.label.show()
        self.writeAssistantRuntimeStatus(
            'running',
            0,
            '检测任务已初始化，正在等待图像采集/处理结果。',
            'detection_initialized',
        )
        msg = {
            'type':'info',
            'data':{
                'tray_id':self.Tray_id
            }
        }
        self.ui_image_queue.put(msg)



        self.myDatabase.addNewTrayToDB()
        # self.detect_button.changeButtonType(0, 'fail')
        # self.detect_button.changeButtonType(1, 'fail')
        # self.detect_button.changeButtonType(2, 'fail')
        # self.detect_button.changeButtonType(2, 'fail')
        # self.debug_ui.scan_xy()
        # self.debug_ui.timestatus = False
        # self.debug_ui.takePhoto()
        # self.debug_ui.autofocus()
        # self.tcp_client.send_message(msg)
        # self..send_message(msg)
        # self.myDatabase.addNewTrayToDB()

    def stop(self):

        msg = {
            'type': 'Control',
            'func': 'Stop'
        }
        self.ui_debug_queue.put(msg)
        self.ui_image_queue.put({
            'type': 'stop_offline_detection',
            'data': {'reason': 'operator_requested'}
        })
        self.stopFlag = True
        # self.debug_ui.stop_flag = True
        self.ui.runBtn.setEnabled(True)
        self.chooseOfflineDirBtn.setEnabled(True)
        self.offlineCheckBox.setEnabled(True)
        self.writeAssistantRuntimeStatus('stopped', 0, '用户已请求停止检测。', 'stopped')


        # sendDataToZYC('func:stop')
        # self.__init__(self.mode)
        # self.stopFlag = True

    def AdminLogin(self):
        print('AdminLogin')

        admin = self.ui.lineEdit_5.text()
        password = self.ui.lineEdit_6.text()
        from .login import admin_password,admin_user_name
        if admin == admin_user_name and password == admin_password:
            self.ui.setBtn.setVisible(True)
            self.ui.label_10.setVisible(True)
            self.ui.setBtn_4.clicked.connect(self.initConfigUI)
            self.ui.info.setText('管理员登录成功！')
            self.ui.stackedWidget_2.setCurrentIndex(0)
            self.ui.stackedWidget_4.setCurrentIndex(0)
            self.setOfflineControlsVisible(True)
        else:
            self.ui.label_29.show()
            thread = threading.Thread(target=lambda: {time.sleep(1), self.ui.label_29.hide()})
            thread.start()
        self.ui.lineEdit_6.setText('')

    def AdminUnlogin(self):
        self.ui.info.setText('管理员退出登录！')
        self.ui.stackedWidget_2.setCurrentIndex(0)
        self.ui.stackedWidget_4.setCurrentIndex(1)
        self.ui.setBtn.setVisible(False)
        self.ui.label_10.setVisible(False)
        self.setOfflineControlsVisible(False)





