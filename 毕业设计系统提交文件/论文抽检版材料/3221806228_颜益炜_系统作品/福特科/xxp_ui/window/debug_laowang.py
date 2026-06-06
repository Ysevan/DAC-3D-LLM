import ctypes
import shutil
import threading
import time
import pandas as pd
import math
from ctypes import memset, cast, c_ubyte
import os
from datetime import datetime
from pathlib import Path
from time import sleep
import queue
import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import gclib
from PyQt5.QtCore import QTimer, QDateTime, QEvent, pyqtSlot, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap, QTransform
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from  duiao import LightSpotAnalyzer
from _ctypes import byref, sizeof, POINTER

from window.swin_t_single import SwinFusion_single

try:
    from window.autofocus_dp.auto_dp import SwinFusionContrastive3, predict
except ImportError:
    from .autofocus_dp.auto_dp import SwinFusionContrastive3, predict

from window.utils import GetInfo
from haizong3 import tenengrad
from window.control.MvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX, MV_SAVE_IMAGE_TO_FILE_PARAM_EX, \
    MV_Image_Bmp
from window.control.camera import CameraFactory, BSLCameraWorker
from zmcdll.tset3 import optimized_tenengrad, fast_circle_crop, detect_circle_and_extract_inside
# from zmcdll.tset3 import detect_circle_and_extract_inside, optimized_tenengrad, fast_circle_crop
from zmcdll.zauxdllPython import ZAUXDLL
from window.control.light import lightControllor
from window.control.getposxy import getposxy

def find_first_local_max_index_numpy(lst):
    """
    在列表中找到第一个极大值（局部最大值）的索引。

    参数:
    lst (list): 输入的数值列表

    返回:
    int: 第一个极大值的索引，如果没有找到极大值则返回 -1
    """
    if len(lst) < 3:
        raise ValueError("列表长度必须至少为3以进行极大值判断")

    for i in range(1, len(lst) - 1):
        if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
            return i

    # 检查列表的第一个和最后一个元素
    if lst[0] > lst[1]:
        return 0
    if lst[-1] > lst[-2]:
        return len(lst) - 1

    return -1  # 如果没有找到极大值
#
def find_first_local_min_index_numpy(lst):
    """
    在列表中找到第一个极小值（局部最小值）的索引。

    参数:
    lst (list): 输入的数值列表

    返回:
    int: 第一个极小值的索引，如果没有找到极小值则返回 -1
    """
    if len(lst) < 3:
        raise ValueError("列表长度必须至少为3以进行极小值判断")

    for i in range(1, len(lst) - 1):
        # 判断是否为局部最小值
        if lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
            return i

    # 检查列表的第一个和最后一个元素
    if lst[0] < lst[1]:
        return 0
    if lst[-1] < lst[-2]:
        return len(lst) - 1

    return -1  # 如果没有找到极小值

class Debug_UI(QtWidgets.QMainWindow):

    timer = QTimer()
    condition = threading.Condition()
    def __init__(self,debug_ui_queue=None,ui_debug_queue=None,debug_image_queue=None,image_debug_queue=None,image_queue = None,mode='mainUICall'):
        super(Debug_UI, self).__init__()
        self.img_flag = None
        self.thread_xy_scan1 = None
        self.auto_focus_sample = None
        self.auto_frame_dict = {}
        self.photo_number = 0
        self.is_showing = False
        self.flag_z_is_bottom = False
        self.thread_run_status = ''
        self.test_mode = 0
        self.pos_z = None
        self.dp_focus_flag = False
        self.debug_image_queue = debug_image_queue
        self.list_pic=[]
        self.image_debug_queue = image_debug_queue
        self.debug_ui_queue = debug_ui_queue
        self.ui_debug_queue = ui_debug_queue
        self.index_auto_focus_sample = 0
        self.autofocus_queue = queue.Queue()
        self.autofocus_flag = False
        self.stop_flag = False
        self.image_queue = image_queue
        self.image_data = {}
        self.pos_y = None
        self.pos_x = None
        self.full_path = None
        self.status = None
        self.pos = None
        self.is_press_down = None
        self.is_press_up = None
        self.camera_save_index = 0

        self.timestatus=True
        self.xy_pos = []
        self.xy_pos = getposxy(-9.2,-28.5)
        self.botton_pos=[]
        self.get_image_flag_temp = 0
        self.auto_frame = []
        self.bottom_pos=[]
        self.is_stopping_z = 0
        self.z_pos_list = []
        self.x_pos_list=[]
        self.is_stopping_x=0
        self.is_stopping_y=0
        self.save_path_layerScan = r'E:\ftkpic\layerscan'
        self.save_path_snakeScan = r'E:\ftkpic\snakescan'
        self.save_path_triggerScan = r'E:\ftkpic\trigger'

        self.layerscan_queue = queue.Queue()

        self.Init_hardware()

        self.step_z = 0
        self.camera_save = 0
        self.area_fit_a = 1527.94
        self.area_fit_b = -11915.09
        self.area_fit_c = 23910.53

        # 焦点参数
        self.laser_focus_z_position =3.96  # 焦点Z位置（mm）
        self.laser_focus_area = 672.50  # 焦点处光斑面积（px²）
        self.laser_focus_cx = 1188.440 # 焦点X质心
        self.laser_focus_cy =658.865  # 焦点Y质心

        # 对焦参数
        self.laser_dead_band_mm = 0.01  # Z位置死区（10μm）
        self.laser_step_um = 10.0  # 每次Z调整步长（μm）
        self.laser_z_range_mm = 2.0  # 允许的Z轴移动范围（±mm）

        # 运行时状态
        self.current_z_position_mm = None  # 当前Z位置（mm）
        self.last_laser_area = None  # 当前光斑面积
        self.last_laser_center = None  # 当前质心 (cx, cy)
        self.is_before_focus = None  # 是否在焦前

        # 定时器
        self.laser_focus_timer = QtCore.QTimer(self)
        self.laser_focus_timer.setInterval(100)  # 100ms
        self.laser_focus_timer.timeout.connect(self.laser_z_adjust_by_spot)
        self.cam2_process_every_n_frames = 10  # 每 10 帧计算一次，可按需调整
        self._cam2_frame_counter = 0
        self._cam2_is_processing = False  # 简易重入保护
        # 按钮绑定
        uic.loadUi(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\ui\debug.ui', self)  # 加载UI文件
        if not mode == 'mainUICall':
            self.show()
        self.btn_save_105.clicked.connect(self.move_x)
        self.btn_save_106.clicked.connect(self.move_y)
        self.btn_save_61.clicked.connect(self.reset_x)
        self.btn_save_66.clicked.connect(self.reset_y)
        self.btn_save_107.clicked.connect(self.move_z)
        self.pushButton_21.clicked.connect(self.stop_z)
        self.pushButton_13.clicked.connect(self.deletepic)
        self.pushButton_2.clicked.connect(self.timestatus_mode)
        # self.pushButton_3.clicked.connect(self.take_10)

        # self.btn_pushButton_12.clicked.connect(self.zcan)
        self.spinBox.valueChanged.connect(self.onValueChanged_cam1)
        self.spinBox_3.valueChanged.connect(self.onValueChanged_cam2)
        self.spinBox_2.valueChanged.connect(self.onValueChanged_cam3)
        self.spinBox_4.valueChanged.connect(self.onValueChanged_cam4)
        self.spinBox_15.valueChanged.connect(self.onValueChanged_Zspeed)

        self.pushButton_24.clicked.connect(self.scan_xy)

        self.btn_save_73.pressed.connect(self.z_up_pressed)
        self.btn_save_73.released.connect(self.z_up_released)
        self.btn_save_72.pressed.connect(self.z_down_pressed)
        self.btn_save_72.released.connect(self.z_down_released)

        self.btn_save_81.pressed.connect(self.z_up_pressed)
        self.btn_save_81.released.connect(self.z_up_released)
        self.btn_save_80.pressed.connect(self.z_down_pressed)
        self.btn_save_80.released.connect(self.z_down_released)

        self.btn_save_77.clicked.connect(self.z_up_10um)
        self.btn_save_78.clicked.connect(self.z_down_10um)

        self.pushButton_26.clicked.connect(self.dp_focus)

        self.pushButton_19.clicked.connect(self.save_pic)
        self.pushButton_18.clicked.connect(self.save_pic)
        self.pushButton_20.clicked.connect(self.save_pic)



        self.pushButton_25.clicked.connect(lambda :{self.autofocus(1)})

        self.btn_save_73.installEventFilter(self)
        self.btn_save_72.installEventFilter(self)
        self.btn_save_81.installEventFilter(self)
        self.btn_save_80.installEventFilter(self)
        self.pushButton_12.clicked.connect(self.zscan)

        self.pushButton_4.clicked.connect(self.auto_focus_pic_take)
        self.pushButton_5.clicked.connect(self.botton_focus)


        self.pushButton_27.clicked.connect(self.on_pushButton_27_clicked)

        self.pushButton.clicked.connect(
            lambda : self.light.light15control(1)
        )
        self.pushButton_16.clicked.connect(
            lambda : self.light.light15control(0)
        )
        self.pushButton_17.clicked.connect(
            lambda: self.light.light60control(1)
        )
        self.pushButton_31.clicked.connect(
            lambda: self.light.light60control(0)
        )
        # 激光
        self.pushButton_6.clicked.connect(
            lambda: self.light.laserpower(0)
        )
        self.pushButton_7.clicked.connect(
            lambda: self.light.laserpower(1)
        )
        self.pushButton_8.clicked.connect(
           self.laser_powere_set
        )

        self.pushButton_3.clicked.connect(self.take10pic)

        self.btn_save_76.clicked.connect(
            # lambda: self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 4)
             lambda:{self.Zmc.ZAux_Direct_SetSpeed(0,1000),self.Zmc.ZAux_Direct_Single_Datum(0,3)}
        )

        self.timer.timeout.connect(self.update_status)
        self.timer.start(100)

        self.getUI_thread = GetInfo(self.ui_debug_queue)
        self.getUI_thread.message_received.connect(self.getInfoFromUI)
        self.getUI_thread.start()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = SwinFusionContrastive3().to(self.device)
        # self.model.load_state_dict(torch.load(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\window\autofocus_dp/best_model.pth'))

        self.model = SwinFusion_single().to(self.device)
        self.model.load_state_dict(
            torch.load(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\bestxinxinxin.pth', weights_only=False)[
                'model_state_dict'])

    @pyqtSlot(dict)
    def getInfoFromUI(self,msg):
        if msg:
            print(msg)

            if msg['type'] == 'Info':
                self.is_showing = not self.is_showing
                if self.is_showing:
                    self.show()
                else:
                    self.hide()
            if msg['type'] == 'Control':
                if msg['func'] == 'Scan':
                    self.scan_xy()

                if msg['func'] == 'Stop':
                    self.stop_flag = True
                if msg['func'] == 'Move':

                    x_pos_start = self.doubleSpinBox_16.value()
                    y_pos_start = self.doubleSpinBox_17.value()
                    self.xy_pos = []
                    self.xy_pos = getposxy(x_pos_start, y_pos_start)
                    button_num = int(msg['button_num']) -1
                    self.g.GCommand(f'pPOSX={ self.xy_pos[button_num][0]}')
                    self.g.GCommand('XQ  # MOVEX,1')
                    self.g.GCommand(f'pPOSY={ self.xy_pos[button_num][1]}')
                    self.g.GCommand('XQ  # MOVEY,2')


    #计算Z位置
    def calculate_z_positions_from_area(self, area):
        """
        根据光斑面积计算两个可能的Z位置（焦前和焦后）
        拟合公式: Area = 317.87*X² - 2394.85*X + 4699.86

        参数:
            area: 当前光斑面积（像素²）
        返回:
            (z1, z2): 两个Z位置（mm），z1 < z2
                     如果无解返回 (None, None)
        """
        a = self.area_fit_a
        b = self.area_fit_b
        c = self.area_fit_c - area

        # 计算判别式
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            print(f"[LASER] 警告：面积 {area:.1f} 超出拟合范围（判别式<0）")
            return None, None

        # 二次方程的两个解
        sqrt_disc = np.sqrt(discriminant)
        z1 = (-b - sqrt_disc) / (2 * a)  # 较小的解（焦前）
        z2 = (-b + sqrt_disc) / (2 * a)  # 较大的解（焦后）

        return z1, z2



    #判断焦前焦后
    def determine_focus_direction(self, cx, cy):
        """
        根据质心位置判断是在焦前还是焦后

        参数:
            cx, cy: 当前质心坐标
        返回:
            True: 焦前
            False: 焦后
        """
        # 计算质心与焦点质心的Y方向偏差
        dy = cy - self.laser_focus_cy

        # 根据Y方向偏移判断
        # 注意：这个判断逻辑可能需要根据实际测试调整
        if dy < 0:
            is_before = True  # 焦前
        else:
            is_before = False  # 焦后

        print(f"[LASER] 质心偏差 dy={dy:.2f} → {'焦前' if is_before else '焦后'}")

        return is_before



    #求光斑质心
    def get_laser_spot_center(self, frame):
        """
        测量光斑，反推Z位置，并判断焦前/焦后

        返回: (cx, cy, area, z_position, is_before_focus) 或 None
        """
        if frame is None:
            return None

        # 转灰度
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 高斯去噪
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu二值化
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 形态学开运算
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 找最大轮廓
        cnt = max(contours, key=cv2.contourArea)

        # 计算光斑面积
        area = cv2.contourArea(cnt)

        if area < 50:
            return None

        # 计算质心
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 反推Z位置并判断方向
        # 1. 计算两个可能的Z位置
        z1, z2 = self.calculate_z_positions_from_area(area)

        if z1 is None:
            return None

        # 2. 根据质心判断是焦前还是焦后
        is_before_focus = self.determine_focus_direction(cx, cy)

        # 3. 选择对应的Z位置
        if is_before_focus:
            z_position = z1  # 焦前（较小的Z）
        else:
            z_position = z2  # 焦后（较大的Z）

        print(f"[LASER] 面积={area:.1f} → Z解: {z1:.3f}mm(焦前), {z2:.3f}mm(焦后) → 选择 {z_position:.3f}mm")

        return cx, cy, area, z_position, is_before_focus

    #Z调整
    def laser_z_adjust_by_spot(self):
        # 检查数据完整性
        if self.current_z_position_mm is None or self.is_before_focus is None:
            print("[LASER] 数据不完整，跳过调整")
            return

        current_z = self.current_z_position_mm
        focus_z = self.laser_focus_z_position

        # 计算偏差（mm）
        delta_z_mm = current_z - focus_z
        delta_z_um = delta_z_mm * 1000

        # 死区判断
        if abs(delta_z_mm) < self.laser_dead_band_mm:
            print(f"[LASER] 已到达焦点 (偏差={delta_z_um:.1f}μm)，停止调整")
            self.laser_focus_timer.stop()
            self.pushButton_27.setText("激光自动对焦")
            return

        # 计算移动步长（不超过实际偏差）
        step_um = min(self.laser_step_um, abs(delta_z_um))
        step_nm = step_um * 1000

        # 判断移动方向
        # delta_z > 0 → 当前在焦后 → 需要下移
        # delta_z < 0 → 当前在焦前 → 需要上移
        if delta_z_mm > 0:
            move_nm = -step_nm
            direction = "下移"
        else:
            move_nm = +step_nm
            direction = "上移"

        # Z轴范围保护
        if abs(delta_z_mm) > self.laser_z_range_mm:
            print(f"[LASER] 偏差过大 ({delta_z_mm:.3f}mm)，超出范围")
            self.laser_focus_timer.stop()
            self.pushButton_27.setText("激光自动对焦")
            return

        # 显示状态信息
        focus_status = "焦前" if self.is_before_focus else "焦后"
        print(
            f"[LASER] {focus_status} | {direction} {step_um:.1f}μm | 当前Z={current_z:.3f}mm | 偏差={delta_z_um:.1f}μm")

        # 执行移动
        self.Zmc.ZAux_Direct_Single_Move(0, move_nm)

    def Init_hardware(self):
        try:

            print('正在初始化硬件')
            self.g = gclib.py()



            self.cam = CameraFactory()
            self.Zmc = ZAUXDLL()
            self.light = lightControllor()

            # print('gclib version:', self.g.GVersion())
            self.g.GOpen('10.0.0.100 --direct -s ALL')
            print('载物台连接成功')

            strtemp = '192.168.0.11'

            iresult = self.Zmc.ZAux_OpenEth(strtemp)
            if 0 != iresult:
                print('运动控制卡连接失败')
            else:
                print('运动控制卡连接成功')
            self.Zmc.ZAux_Direct_SetAtype(0, 1)  # 类型：方向+脉冲
            self.Zmc.ZAux_Direct_SetUnits(0, 40)  # 1units=40脉冲，1um
            # print('脉冲当量',self.Zmc.ZAux_Direct_GetUnits)
            self.Zmc.ZAux_Direct_SetSpeed(0, 1000)
            self.Zmc.ZAux_Direct_SetCreep(0, 1000)
            # time.sleep(1)
            self.Zmc.ZAux_Direct_SetDatumIn(0,0)
            #

            camList = self.cam.setFunc(self.camera_get_img)  # 得到三相机cam句柄
            custom_order = {"DA3827093": 0, "DA3562117": 1, "DA3562103": 2,"DA6630523": 3}
            # custom_order = {"DA3827093": 0, "DA3562117": 1, "DA3562103": 2}

            # 使用 sorted 函数，并使用 lambda 表达式作为 key 参数
            self.camList = sorted(camList, key=lambda camList: custom_order[camList['number']])
            print('四相机连接成功')
            self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", 40000)
            self.camList[1]['cam'].MV_CC_SetFloatValue("ExposureTime", 20000)
            self.camList[2]['cam'].MV_CC_SetFloatValue("ExposureTime", 6000)
            self.camList[3]['cam'].MV_CC_SetFloatValue("ExposureTime", 100)

            # self.bslworker = BSLCameraWorker('40230735',1)
            # self.bsl_thread = QThread()
            # self.bslworker.moveToThread(self.bsl_thread)
            # self.bslworker.frame_ready.connect(self.camera_get_img)  # 接收图像帧
            # self.bsl_thread.started.connect(self.bslworker.run)
            # self.bsl_thread.start()

            if self.debug_ui_queue:
                msg = {
                    'type':'hardware_info',
                    'data':'hardware is finished'
                }
                self.debug_ui_queue.put(msg)
        except Exception as e:
            print(f'硬件初始化出现错误：{e}')
# 蛇形扫描
    def scan_xy(self):
        delaytime=1

        self.save_path_snakeScan = 'E:/ftkpic/snakescan/'+self.date_folder + '-' + self.time_folder
        os.makedirs(self.save_path_snakeScan, exist_ok=True)
        os.makedirs(self.save_path_snakeScan+'/焦前', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦后', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦面', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/检测结果', exist_ok=True)

        x_pos_start = self.doubleSpinBox_16.value()
        y_pos_start = self.doubleSpinBox_17.value()
        self.xy_pos = []
        self.xy_pos = getposxy(x_pos_start,y_pos_start)


        self.thread_xy_scan = threading.Thread(target=self.xy_scan_thread, args=(self.xy_pos, ))
        self.thread_xy_scan.start()


        with self.condition:
            self.thread_run_status = 'xy'
            self.condition.notify_all()

    def xy_scan_thread(self, xy_pos):
        # print('启动xy线程')
        with self.condition:
            self.condition.wait_for(lambda: self.thread_run_status == 'xy')
            # print('执行xy')

            # try:
            start_time = time.time()
            self.timestatus = False
            if self.debug_ui_queue:
                msg = {
                    'type':'info',
                    'data':{
                        'start_time':start_time
                    }
                }
                self.debug_ui_queue.put(msg)

            self.status = 'xy_scan'
            # self.g.GCommand(f'pPOSY={xy_pos[0][1]}')
            # self.g.GCommand('XQ  # MOVEY,2')
            # self.xy_status()
            index = 0
            for (i, pos) in enumerate(xy_pos):
            #     y_now = xy_pos[0][1]

                if i % 12 == 0:
                    y_now = xy_pos[i][1]
                    # print('y:',y_now)
                    print('y位置',i%2)
                    index += 1
                    self.g.GCommand(f'pPOSY={y_now}')
                    self.g.GCommand('XQ  # MOVEY,2')
                    self.xy_status()
                    self.get_image_flag_temp = 0
                    res_x = self.g.GCommand('TPA')
                    res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    res_y = self.g.GCommand('TPB')
                    res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    if res_x.strip():
                        # self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
                        self.pos_x = round(float(res_x)) / 6400
                    if res_y.strip():
                        # self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
                        self.pos_y = round(float(res_y)) / 6400
                    self.autofocus(i%2)
                    self.condition.wait_for(lambda: self.thread_run_status == 'xy')
                else:
                    x = pos[0]
                    # print('x:',x)
                    print('x位置:',i%2)
                    index += 1
                    self.g.GCommand(f'pPOSX={x}')
                    self.g.GCommand('XQ  # MOVEX,1')
                    # self.stagestatus_x()
                    self.xy_status()

                    self.get_image_flag_temp = 0
                    res_x = self.g.GCommand('TPA')
                    res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    res_y = self.g.GCommand('TPB')
                    res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    if res_x.strip():
                        # self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
                        self.pos_x = round(float(res_x)) / 6400
                    if res_y.strip():
                        # self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
                        self.pos_y = round(float(res_y)) / 6400
                    self.autofocus(i%2)
                    self.condition.wait_for(lambda: self.thread_run_status == 'xy')
                    # time.sleep(1)
                print(index)
                # print(x, y_now)



            self.timestatus = True
            end_time = time.time()
            print(f"蛇形扫描用时: {end_time - start_time} seconds")
            # except Exception as e:
            #     print(e)

    def take10pic(self):
        for i in range(10):
            # 使用QTimer延迟执行，避免界面冻结
            QTimer.singleShot(i * 1000, self.pushButton_20.click)

  # 底面对焦
    def botton_focus(self):
        self.save_path_snakeScan = 'E:/ftkpic/snakescan/' + self.date_folder + '-' + self.time_folder
        os.makedirs(self.save_path_snakeScan, exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦前', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦后', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦面', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/检测结果', exist_ok=True)

        x_pos_start = self.doubleSpinBox_16.value()
        y_pos_start = self.doubleSpinBox_17.value()
        self.xy_pos = []
        self.xy_pos = getposxy(x_pos_start, y_pos_start)

        self.thread_xy_scan1 = threading.Thread(target=self.xy_scan1_thread, args=(self.xy_pos,))
        self.thread_xy_scan1.start()

        with self.condition:
            self.thread_run_status = 'xy'
            self.condition.notify_all()

    # def xy_scan1_thread(self, xy_pos):
    #     # print('启动xy线程')
    #     with self.condition:
    #         self.condition.wait_for(lambda: self.thread_run_status == 'xy')
    #         # print('执行xy')
    #
    #         # try:
    #         start_time = time.time()
    #         self.timestatus = False
    #         if self.debug_ui_queue:
    #             msg = {
    #                 'type': 'info',
    #                 'data': {
    #                     'start_time': start_time
    #                 }
    #             }
    #             self.debug_ui_queue.put(msg)
    #
    #         # self.status = 'xy_scan'
    #         self.status ='snake_scan'
    #         df = pd.read_csv(r"C:\Users\Administrator\Desktop\工装盘焦面11111111.csv", header=None,encoding="gbk")#文件编码类型
    #         b_column_data = df.iloc[:, 2].tolist()
    #         index = 0
    #         for (i, pos) in enumerate(xy_pos):
    #             #     y_now = xy_pos[0][1]
    #
    #             if i % 12 == 0:
    #                 y_now = xy_pos[i][1]
    #                 # print('y:',y_now)
    #                 print('y位置', i % 2)
    #                 index += 1
    #                 self.g.GCommand(f'pPOSY={y_now}')
    #                 self.g.GCommand('XQ  # MOVEY,2')
    #                 time.sleep(5)
    #                 self.xy_status()
    #                 self.get_image_flag_temp = 0
    #                 # res_x = self.g.GCommand('TPA')
    #                 # res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
    #                 # res_y = self.g.GCommand('TPB')
    #                 # res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
    #                 # if res_x.strip():
    #                 #     self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
    #                 #     self.pos_x = round(float(res_x)) / 6400
    #                 # if res_y.strip():
    #                 #     self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
    #                 #     self.pos_y = round(float(res_y)) / 6400
    #                 self.move_z_up(b_column_data[i])
    #                 self.condition.wait_for(lambda: self.thread_run_status == 'xy')
    #                 # self.takePhoto()
    #                 self.camera_save = 4
    #                 self.status ='snake_scan'
    #                 self.takePhoto()
    #
    #             else:
    #                 x = pos[0]
    #                 # print('x:',x)
    #                 print('x位置:', i % 2)
    #                 index += 1
    #                 self.g.GCommand(f'pPOSX={x}')
    #                 self.g.GCommand('XQ  # MOVEX,1')
    #                 time.sleep(5)
    #                 # self.stagestatus_x()
    #                 self.xy_status()
    #
    #                 self.get_image_flag_temp = 0
    #                 # res_x = self.g.GCommand('TPA')
    #                 # res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
    #                 # res_y = self.g.GCommand('TPB')
    #                 # res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
    #                 # if res_x.strip():
    #                 #     self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
    #                 #     self.pos_x = round(float(res_x)) / 6400
    #                 # if res_y.strip():
    #                 #     self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
    #                 #     self.pos_y = round(float(res_y)) / 6400
    #                 self.move_z_up(b_column_data[i])
    #                 self.condition.wait_for(lambda: self.thread_run_status == 'xy')
    #                 # self.takePhoto()
    #                 # self.camera_save = 1
    #                 #激光控制
    #                 # 开激光
    #                 self.light.laserpower(0)
    #                 # Z轴补偿移动（基于cam2的结果）
    #                 #a = 0 - self.last_result_cam2[2]
    #                 #self.Zmc.ZAux_Direct_Single_Move(0, a * 1000)
    #                 # 等待线程状态确认
    #                 self.condition.wait_for(lambda: self.thread_run_status == 'xy')
    #                 time.sleep(0.5)
    #                 # 检查Z轴状态
    #                 self.z_status()
    #                 time.sleep(0.5)
    #
    #                 # 关激光
    #                 self.light.laserpower(1)
    #                 #相机模式存图
    #                 self.camera_save = 4
    #                 self.status = 'snake_scan'
    #                 self.takePhoto()
    #
    #             print(index)
    def xy_scan1_thread(self, xy_pos):
        # print('启动xy线程')
        with self.condition:
            self.condition.wait_for(lambda: self.thread_run_status == 'xy')

            start_time = time.time()
            self.timestatus = False

            if self.debug_ui_queue:
                msg = {'type': 'info', 'data': {'start_time': start_time}}
                self.debug_ui_queue.put(msg)

            self.status = 'snake_scan'
            #存激光图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            laser_save_dir = rf"E:\ftkpic\激光扫描图_{timestamp}"
            os.makedirs(laser_save_dir, exist_ok=True)
            print(f"激光图将保存至：{laser_save_dir}")

            # 读取CSV (Z轴数据)
            # 建议加上 dtype 转换防止混入字符串导致崩溃
            df = pd.read_csv(r"C:\Users\Administrator\Desktop\工装盘焦面11111111.csv", header=None, encoding="gbk")
            b_column_data = df.iloc[:, 2].tolist()
            index = 0

            for (i, pos) in enumerate(xy_pos):


                if i % 12 == 0:
                    # === Y轴移动 (换行) ===
                    y_now = xy_pos[i][1]
                    print(f'Y移动目标: {y_now}, 索引: {index}')
                    index += 1
                    self.g.GCommand(f'pPOSY={y_now}')
                    self.g.GCommand('XQ  # MOVEY,2')
                else:
                    # === X轴移动 (行内) ===
                    x = pos[0]
                    print(f'X移动目标: {x}, 索引: {index}')
                    index += 1
                    self.g.GCommand(f'pPOSX={x}')
                    self.g.GCommand('XQ  # MOVEX,1')


                time.sleep(0.05)
                self.xy_status()  # 确保此函数能阻塞直到电机停止
                self.get_image_flag_temp = 0


                self.condition.wait_for(lambda: self.thread_run_status == 'xy')

                self.light.laserpower(0)  # 0 = 开激光

                time.sleep(1)  # 按照要求：停1秒

                # --- 第一次拍照 (激光图) ---
                self.camera_save = 5
                self.status = 'snake_scan'

                # 这里保留你原有的第二个延时，如果觉得太慢可以去掉
                time.sleep(1)

                self.takePhoto()  # 拍激光图

                # 【新增】拍照后给一点缓冲时间，确保图片数据传输完成
                time.sleep(0.5)


                self.light.laserpower(1)  # 1 = 关激光
                # print("激光已关闭，准备移动Z轴")


                target_z = b_column_data[i]
                self.move_z_up(target_z)


                self.condition.wait_for(lambda: self.thread_run_status == 'xy')
                time.sleep(0.05)  # 短暂缓冲确保指令送达
                self.z_status()  # 检查Z轴是否停止


                self.camera_save = 4  # 恢复为 4 (保存到检测结果或焦面)
                self.status = 'snake_scan'
                self.takePhoto()  # 拍无激光的图

                print(f"Index: {index} 单点流程结束")

    # 自动对焦
    def autofocus(self,mode):

        self.timestatus = False
        self.autofocus_flag = True
        self.flag_z_is_bottom = None
        self.z_pos_list=[]

        self.tenengrad_list = []
        print('mode:', mode)
        self.take_img_thread = threading.Thread(target=self.takeAutoFocusImg)
        self.move_thread = threading.Thread(target=lambda :self.AutoMove(mode))
        self.get_img_thread = threading.Thread(target=self.getAutoFocusImg)

        self.move_thread.start()
        self.take_img_thread.start()
        self.get_img_thread.start()

        with self.condition:
            self.thread_run_status = 'auto'
            self.condition.notify_all()

    def AutoMove(self,mode):
        # print('启动auto线程')
        with self.condition:
            self.condition.wait_for(lambda: self.thread_run_status == 'auto')
            # print('auto执行')

            if not mode:
                self.pos_top = self.doubleSpinBox_2.value()
                self.pos_bottom =self.doubleSpinBox.value()
            else:
                self.pos_bottom = self.doubleSpinBox_2.value()
                self.pos_top = self.doubleSpinBox.value()

            print(self.pos_top,self.pos_bottom)

            # self.pos_top = 7.5
            # self.pos_bottom = 8.5

            self.auto_start_time = time.time()
            self.move_z_up(self.pos_top)
            self.z_status()
            # while True:
            #     self.Zmc.ZAux_Direct_SetUserVar('g_cmd',6)
            #     is_stopping = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
            #     print(is_stopping)
            #     if is_stopping==0:
            #         break
            # print(str(pos_z) == str(pos_top))

            # self.cam.setMode(0)
            self.thread_run_status = 'take_pic'
            self.condition.notify_all()
            self.move_z_up(self.pos_bottom)


            # self.condition.wait_for(lambda :self.thread_run_status == 'auto')
    #按钮激光
    def on_pushButton_27_clicked(self):  # 或你实际的函数名
        """
        启动/停止激光自动对焦
        """
        if self.laser_focus_timer.isActive():
            # 停止对焦
            self.laser_focus_timer.stop()
            print("[LASER] 自动对焦已停止")
            self.pushButton_27.setText("激光自动对焦")
        else:
            # 启动对焦
            self.laser_focus_timer.start()
            print("[LASER] 自动对焦已启动")
            self.pushButton_27.setText("停止激光对焦")

    #Z轴调整
    # def laser_z_adjust(self):


        # # 1. 有效性判断
        # if not self.laser_x_valid or self.laser_x_est is None:
        #     return
        #
        # x_curr = self.laser_x_est
        # x_focus = self.laser_focus_x
        # error = x_curr - x_focus
        #
        # # 2. 焦面死区（防抖）
        # DEAD_BAND = 0.5  # mm
        #
        # if abs(error) < DEAD_BAND:
        #     return  # 已在焦面附近，不动
        #
        # # 3. 微调步长（先小）
        # STEP = 0.1  # mm
        #
        # if error < 0:
        #     # 焦前 → Z 上移
        #     self.Zmc.ZAux_Direct_Single_Move(0, +STEP * 1000)
        # else:
        #     # 焦后 → Z 下移
        #     self.Zmc.ZAux_Direct_Single_Move(0, -STEP * 1000)

    def takeAutoFocusImg(self):
        # print('启动图像线程')
        with self.condition:
            self.condition.wait_for(lambda: self.thread_run_status == 'take_pic')
            self.flag_z_is_bottom = False
        index=0
        while not self.flag_z_is_bottom:
            # print('拍摄图像')
            self.takePhoto()
            index+=1
            print(index)
            time.sleep(0.05)

    def getAutoFocusImg(self):
        data_list = [] #存储队列传来的消息
        tenengrad_list = []
        while True:
            try:
                item = self.autofocus_queue.get()
                if item:
                    if isinstance(item,bool):
                        self.end_auto_time = time.time()
                        print(f'自动对焦用时：{self.end_auto_time - self.auto_start_time} seconds')
                        max_index = find_first_local_max_index_numpy(tenengrad_list)
                        # print(f'最大：',max_index)
                        print(f'最大：', data_list[max_index].items())
                        print(f'焦面位置：{self.z_pos_list[max_index]}')
                        self.bottom_pos.append(self.z_pos_list[max_index])
                        with open("bottom_pos.txt", "a", encoding="utf-8") as file:
                            file.write(str(self.z_pos_list[max_index]))
                            # for item in tenengrad_list:
                            #     file.write(str(item) + "\t")
                            file.write("\n")
                        for key, value in data_list[max_index].items():
                            cv2.imwrite(f'{key}.png',value)
                            if key == 'DA3562103':
                                pic_path = rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp"
                            if key == 'DA3562117':
                                pic_path = rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp"
                            if key == 'DA3827093':
                                pic_path = rf"{self.save_path_snakeScan}\焦面\{self.pos_x},{self.pos_y}.bmp"

                            self.image_data[key] = {
                                'frame': value,
                                'pic_path': pic_path,
                                # 'result_pic_path':result_pic_path
                            }
                        if self.debug_image_queue:
                            result_pic_path = rf"{self.save_path_snakeScan}\检测结果\{self.pos_x},{self.pos_y}.jpg"
                            self.image_data['result_pic_path'] = result_pic_path
                            self.debug_image_queue.put(self.image_data)

                            self.image_data = {}

                        with self.condition:
                            self.thread_run_status = 'xy'
                            self.condition.notify_all()
                        break
                    else:
                        data_list.append(item)
                        value = tenengrad(detect_circle_and_extract_inside(item['DA3827093'])[0])
                        # value=optimized_tenengrad(fast_circle_crop(item['DA3827093'])[0])
                        # value = tenengrad(item['DA3827093'])
                        tenengrad_list.append(value)
            except queue.Empty:
                pass

        # print(self.tenengrad_list)
        # pos = []
        # value = []
        # start = time.time()
        # for item in self.tenengrad_list:
        #     pos.append(item[1])
        #     value.append(tenengrad(item[0]))
        #
        # max_index = find_first_local_max_index_numpy(value)
        # end = time.time()
        # print(f'{end-start}s')
        # max_pos = pos[max_index]
        # print(max_pos)
        # self.move_z_up(max_pos)
        # self.z_status()
        # self.takePhoto()
        # self.end_auto_time = time.time()
        # print(f'自动对焦用时：{self.end_auto_time - self.auto_start_time} seconds')
        # #
        # pos = self.tenengrad_list[1][max_index]
        # print(pos)
        #     print('图像执行')
        #     clarity_tenengrad_list = []
        #     pos_list =[]
        #     print(self.autofocus_queue.empty())
        #     while not self.autofocus_queue.empty():
        #         try:
        #             self.autofocus_queue.get_nowait()  # 非阻塞方式
        #         except queue.Empty:
        #             break
        #     self.autofocus_flag = True
        #     self.thread_run_status = 'auto'
        #     self.condition.notify_all()
        #     print(self.thread_run_status)
        #
        #
        #
        # while True:
        #
        #     try:
        #         data = self.autofocus_queue.get(timeout=5)
        #         # print(data)
        #         frame = data[0]
        #         pos = data[1]
        #         if isinstance(frame, np.ndarray):
        #             # print(frame.shape,pos)
        #             # print(f'自动对焦图像：高{frame.shape[0]},宽{frame.shape[1]}')
        #             clarity_tenengrad_list.append(tenengrad(frame))
        #             pos_list.append(pos)
        #
        #
        #
        #
        #             print(pos,self.pos_bottom)
        #             if pos == self.pos_bottom:
        #                 print('到达底部')
        #                 # max_index = clarity_tenengrad_list.index(max(clarity_tenengrad_list))
        #                 if len(clarity_tenengrad_list)<3:
        #                     continue
        #                 print('自动对焦获取图像数量：',len(clarity_tenengrad_list), len(pos_list))
        #                 max_index = find_first_local_max_index_numpy(clarity_tenengrad_list)
        #                 max_pos = pos_list[max_index]
        #                 print(max_pos)
        #                 self.move_z_up(max_pos)
        #                 # min_index = find_first_local_min_index_numpy(clarity_tenengrad_list)
        #                 # min_pos = pos_list[min_index]
        #                 # print(min_pos)
        #                 self.auto_end_time = time.time()
        #                 print(f"自动对焦用时: {self.auto_end_time - self.auto_start_time} seconds")
        #                 clarity_tenengrad_list = []
        #                 pos_list = []
        #                 self.auto_start_time = 0
        #                 self.auto_end_time = 0
        #                 pos = 0
        #                 frame = None
        #                 # self.timestatus = True
        #                 self.autofocus_flag=False
        #                 while not self.autofocus_queue.empty():
        #                     try:
        #                         self.autofocus_queue.get_nowait()  # 非阻塞方式
        #                     except queue.Empty:
        #                         break
        #
        #                 break
        #     except Exception as e:
        #         print('chaoshi')
        # print('结束线程')
            # clarity_tenengrad_list = []
            # self.auto_end_time = time.time()
            # print(f"自动对焦用时: {self.auto_end_time - self.auto_start_time} seconds")
            # self.auto_end_time = 0
            # self.auto_start_time = 0
            # break

#激光
    def laser_powere_set(self):
        power = self.doubleSpinBox_18.value()
        self.light.laserpower1(int(power))

#Z轴控制
    def z_up_pressed(self):
        self.Zmc.ZAux_Direct_Single_Vmove(0,1)

    def z_up_released(self):
        self.Zmc.ZAux_Direct_Single_Cancel(0,3)

    def z_down_pressed(self):
        self.Zmc.ZAux_Direct_Single_Vmove(0,-1)

    def z_down_released(self):
        self.Zmc.ZAux_Direct_Single_Cancel(0,3)

    def move_z_relative_down(self):

        while True:
            if self.is_press_down:
                self.Zmc.ZAux_Direct_Single_Move(0, -10)
                # self.Zmc.ZAux_Direct_SetUserVar('relative_step', -1000)
                # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
                sleep(0.05)
            else:
                # self.Zmc.ZAux_Direct_Rapidstop(0)
                break

    def stop_z(self):
        # self.Zmc.ZAux_Direct_Single_Move(0, 0)
        self.Zmc.ZAux_Direct_Rapidstop(0)
        # self.Zmc.ZAux_Direct_SetUserVar('relative_step', 0)
        # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
# Z位置
    def get_z_pos(self):
        ret, value = self.Zmc.ZAux_Direct_GetMpos(0)  # 假设轴号为0
        if ret == 0:  # 检查错误码是否为0（成功）
            end_position = value.value * 40 / 40000  # 通过.value读取实际值
            return end_position
            # print(f"最终位置: {end_position}")
        else:
            print(f"错误码: {ret}")

# Z轴绝对运动
    def move_z_up(self,pos):
        ret = self.Zmc.ZAux_Direct_Single_MoveAbs(0,pos*1000)
        if not ret == 0:
            print('z轴绝对运动报错！！！！！！！！')
        # pos=pos*40000
        # self.Zmc.ZAux_Direct_SetUserVar('absolute_step', pos)
        # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 2)
# Z轴速度
    def onValueChanged_Zspeed(self, val):
        # speed = val * 600 / 1000
        self.Zmc. ZAux_Direct_SetSpeed(0,val)
        # self.Zmc.ZAux_Direct_SetUserVar('z_speed', speed)
        # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 5)

    def move_z(self):
        # print("2222")
        a = self.doubleSpinBox_12.value()
        self.move_z_up(a)
#    层扫
    def zscan(self):
        self.img_flag = 'layerscan'
        scan_up=self.doubleSpinBox_9.value()
        scan_down=self.doubleSpinBox_14.value()
        scan_step=self.doubleSpinBox_13.value()
        base_path = "E:/ftkpic/layerscan"  # 指定基础路径
        date_folder_path = os.path.join(base_path, self.date_folder)
        os.makedirs(date_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略

        # 创建时间文件夹
        time_folder_path = os.path.join(date_folder_path, self.time_folder)
        os.makedirs(time_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略

        self.full_path = time_folder_path

        # self.move_z_up(scan_up)

        receive_img_thread = threading.Thread(target=self.receive_img)
        thread_z_scan = threading.Thread(target=self.z_scan_thread,args=(scan_up,scan_down,scan_step,))
        thread_z_scan.start()
        receive_img_thread.start()

    def auto_focus_pic_take(self):
        self.auto_focus_sample = True
        self.focus_pos = self.doubleSpinBox_15.value()
        scan_step = 0.02
        scan_up = self.focus_pos + 50 * scan_step
        scan_down = self.focus_pos - 50 * scan_step

        step = 0.02
        start = 1.0
        end = -1.0
        num_points = int((start - end) / step) + 1
        self.auto_image_name = [round(start - i * step, 2) for i in range(num_points)]
        # self.auto_image_name = np.arange(scan_up,)

        base_path = "E:/ftkpic/auto_focus_sample"  # 指定基础路径
        os.makedirs(base_path, exist_ok=True)  # 如果文件夹已存在则忽略

        sample_dir = f'{len([p for p in Path(base_path).iterdir() if p.is_dir()])+1}'
        sample_path = os.path.join(base_path, sample_dir)
        os.makedirs(sample_path, exist_ok=True)  # 如果文件夹已存在则忽略

        # sub_dir = os.path.join(sample_path, self.time_folder)
        os.makedirs(sample_path+'/cam1', exist_ok=True)
        os.makedirs(sample_path + '/cam2', exist_ok=True)
        os.makedirs(sample_path + '/cam3', exist_ok=True)


        self.cam1_path = sample_path+'/cam1'
        self.cam2_path = sample_path + '/cam2'
        self.cam3_path = sample_path + '/cam3'




        thread_z_scan = threading.Thread(target=self.z_scan_thread1, args=(scan_up, scan_down, scan_step,))
        thread_z_scan.start()

    def z_scan_thread1(self, scan_up, scan_down, scan_step):
        self.timestatus = False
        self.status = 'take_sample_pic'
        self.move_z_up(scan_up)
        self.z_status()

        self.pos_z = round(self.get_z_pos(), 4)
        # print(self.pos_z)
        # time.sleep(2)
        self.camera_save = 5
        self.status = 'take_sample_pic'
        self.takePhoto()
        while self.pos_z > scan_down:
            self.move_z_up(self.pos_z - scan_step)
            self.z_status()
            # sleep(1)
            # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
            # self.pos_z = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
            self.pos_z = self.get_z_pos()
            # print(self.pos_z)
            self.camera_save = 5
            self.status = 'take_sample_pic'
            self.takePhoto()

        # self.timestatus = True

    def dp_focus(self):
        self.dp_focus_img_list = {}
        self.dp_focus_flag = True
        # self.dp_img_queue = Queue(3)
        self.takePhoto()
        thread = threading.Thread(target=self.dp_focus_get_img)
        thread.start()

    def dp_focus_get_img(self):
        while True:
            if len(self.dp_focus_img_list) >= 3:
                print(self.dp_focus_img_list)
                im1 = self.dp_focus_img_list['DA3827093']
                im2 = self.dp_focus_img_list['DA3562117']
                im3 = self.dp_focus_img_list['DA3562103']
                res = predict(im1,im2,im3,self.model,self.device)
                print(res)

                self.Zmc.ZAux_Direct_Single_Move(0,-res*1000)
                self.z_status()
                self.dp_focus_flag = False
                self.dp_focus_img_list = {}
                self.takePhoto()
                break


    # 老王
    def laowang(self):

        # for i in range(10):
        #     print(self.camList[0]['number'] + f'调整曝光时间：{40000}')
        #     ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(40000))
        #     time.sleep(5)
        #     print(self.camList[0]['number'] + f'调整曝光时间：{400}')
        #     ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(400))


        self.save_path_snakeScan = 'E:/ftkpic/snakescan/' + self.date_folder + '-' + self.time_folder
        os.makedirs(self.save_path_snakeScan, exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦前', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦后', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦面', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/检测结果', exist_ok=True)

        x_pos_start = self.doubleSpinBox_16.value()
        y_pos_start = self.doubleSpinBox_17.value()
        self.xy_pos = []
        self.xy_pos = getposxy(x_pos_start, y_pos_start)

        self.thread_xy_scan2 = threading.Thread(target=self.xy_scan2_thread, args=(self.xy_pos,))
        self.thread_xy_scan2.start()

        with self.condition:
            self.thread_run_status = 'xy'
            self.condition.notify_all()

    def xy_scan2_thread(self, xy_pos):
        # print('启动xy线程')
        with self.condition:
            self.condition.wait_for(lambda: self.thread_run_status == 'xy')
            # print('执行xy')

            # try:
            start_time = time.time()
            self.timestatus = False
            if self.debug_ui_queue:
                msg = {
                    'type': 'info',
                    'data': {
                        'start_time': start_time
                    }
                }
                self.debug_ui_queue.put(msg)

            # self.status = 'xy_scan'
            self.status ='snake_scan'

            index = 0
            for (i, pos) in enumerate(xy_pos):
                #     y_now = xy_pos[0][1]

                if i % 12 == 0:
                    y_now = xy_pos[i][1]
                    # print('y:',y_now)
                    print('y位置', i % 2)
                    index += 1
                    self.g.GCommand(f'pPOSY={y_now}')
                    self.g.GCommand('XQ  # MOVEY,2')
                    self.xy_status()
                    self.get_image_flag_temp = 0
                    # res_x = self.g.GCommand('TPA')
                    # res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    # res_y = self.g.GCommand('TPB')
                    # res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    # if res_x.strip():
                    #     self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
                    #     self.pos_x = round(float(res_x)) / 6400
                    # if res_y.strip():
                    #     self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
                    #     self.pos_y = round(float(res_y)) / 6400
                    # self.move_z_up(b_column_data[i])

                    # 调整曝光400
                    # print(self.camList[0]['number']+f'调整曝光时间：{400}')
                    # ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(400))
                    # 开激光
                    self.light.laserpower(0)
                    #a = 0 - self.last_result_cam2[2]
                    #self.Zmc.ZAux_Direct_Single_Move(0, a * 1000)
                    self.condition.wait_for(lambda: self.thread_run_status == 'xy')
                    # self.takePhoto()
                    self.z_status()
                    # 关激光
                    self.light.laserpower(1)
                    # 调整曝光400
                    # print(self.camList[0]['number'] + f'调整曝光时间：{40000}')
                    # ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(40000))
                    # time.sleep(0.1)
                    self.camera_save = 4#蛇扫模式
                    self.status ='snake_scan'
                    self.takePhoto()


                else:
                    x = pos[0]
                    # print('x:',x)
                    print('x位置:', i % 2)
                    index += 1
                    self.g.GCommand(f'pPOSX={x}')
                    self.g.GCommand('XQ  # MOVEX,1')
                    # self.stagestatus_x()
                    self.xy_status()

                    self.get_image_flag_temp = 0
                    # res_x = self.g.GCommand('TPA')
                    # res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    # res_y = self.g.GCommand('TPB')
                    # res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    # if res_x.strip():
                    #     self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
                    #     self.pos_x = round(float(res_x)) / 6400
                    # if res_y.strip():
                    #     self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
                    #     self.pos_y = round(float(res_y)) / 6400
                    # self.move_z_up(b_column_data[i])
                    # 调整曝光400
                    # print(self.camList[0]['number'] + f'调整曝光时间：{400}')
                    # ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(400))
                    # time.sleep(0.1)
                    # 开激光
                    self.light.laserpower(0)

                    # a = 0 - self.last_result_cam3[2]
                    # self.Zmc.ZAux_Direct_Single_Move(0, a * 1000)
                    self.condition.wait_for(lambda: self.thread_run_status == 'xy')
                    # self.takePhoto()
                    self.z_status()
                    # 关激光
                    self.light.laserpower(1)
                    # 调整曝光400
                    # print(self.camList[0]['number'] + f'调整曝光时间：{40000}')
                    # ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(40000))
                    # time.sleep(0.1)
                    # self.camera_save = 1
                    self.camera_save = 4
                    self.status = 'snake_scan'
                    self.takePhoto()

                    # time.sleep(1)
                print(index)









    def z_status(self):

        self.is_stopping_z=self.Zmc.ZAux_Direct_GetIfIdle(0)[1].value
        # print(self.is_stopping_z)
        while self.is_stopping_z==0:
            self.is_stopping_z=self.Zmc.ZAux_Direct_GetIfIdle(0)[1].value
            # print('z轴没停止',self.is_stopping_z)
        time.sleep(0.01)

    def z_scan_thread(self,scan_up,scan_down,scan_step):
        self.timestatus = False
        self.status = 'z_scan'
        # self.Zmc.ZAux_Direct_SetUserVar('z_speed', 500)
        # self.Zmc.ZAux_Direct_SetUserVar('z_speed', (scan_up-scan_down)*100/5)
        # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 5)
        # print('z_scan_thread')
        # sleep(0.5)
        self.move_z_up(scan_up)
        self.z_status()

        self.pos_z=round(self.get_z_pos(),4)
        # print(self.pos_z)
        self.camera_save = 1
        self.status = 'z_scan'
        self.takePhoto()
        while self.pos_z>scan_down:
            self.move_z_up(self.pos_z-scan_step)
            self.z_status()
            # sleep(1)
            # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
            # self.pos_z = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
            self.pos_z=round(self.get_z_pos(),4)
            # print(self.pos_z)
            self.camera_save = 3
            self.status = 'z_scan'
            self.takePhoto()

        self.img_flag = None
        # self.timestatus = True

    def z_up_10um(self):
        self.Zmc.ZAux_Direct_Single_Move(0,10)
    def z_down_10um(self):
        self.Zmc.ZAux_Direct_Single_Move(0,-10)
    # 图片删除
    def deletepic(self):
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder_path:
            return  # 如果用户取消选择，直接返回

        # 确认是否删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除文件夹 '{folder_path}' 下的所有内容吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return  # 如果用户选择不删除，直接返回

        # 删除文件夹内容
        try:
            self.delete_folder_contents(folder_path)
            QMessageBox.information(self, "完成", "文件夹内容已删除！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")

    def delete_folder_contents(self, folder_path):
        # 遍历文件夹内容并删除
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if (os.path.isfile(item_path)
                    or os.path.islink(item_path)):
                os.unlink(item_path)  # 删除文件或符号链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除子文件夹及其内容
#相机控制
    def onValueChanged_cam1(self,val):
        print(self.camList[0]['number']+f'调整曝光时间：{val}')
        ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))

    def onValueChanged_cam2(self,val):
        print(self.camList[1]['number'] + f'调整曝光时间：{val}')
        ret = self.camList[1]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))

    def onValueChanged_cam3(self, val):
        print(self.camList[2]['number'] + f'调整曝光时间：{val}')
        ret = self.camList[2]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))

    def onValueChanged_cam4(self, val):
        print(self.camList[3]['number'] + f'调整曝光时间：{val}')
        ret = self.camList[3]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))
    def takePhoto(self):

        self.Zmc.ZAux_Direct_SetOp(0, 1)
        # self.Zmc.ZAux_Direct_SetOp(1, 1)
        # self.Zmc.ZAux_Direct_SetOp(2, 1)
        sleep(0.05)
        self.Zmc.ZAux_Direct_SetOp(0, 0)
        # self.Zmc.ZAux_Direct_SetOp(1, 0)
        # self.Zmc.ZAux_Direct_SetOp(2, 0)
        # print('picture')
# 存图

    def save_pic(self):
        button = self.sender()
        print(button.objectName())
        self.status = 'trigger'
        if button.objectName() == 'pushButton_20':  # 相机1触发存图
            self.camera_save = 1
        if button.objectName() == 'pushButton_18':  # 相机2触发存图
            self.camera_save = 2
        if button.objectName() == 'pushButton_19':  # 相机3触发存图
            self.camera_save = 3
        # if button.objectName() == 'pushButton_3':  # 相机1触发10存图
        #     # for i in range(10):
        #     #     self.camera_save = 1
        #     #     self.takePhoto()
        #     #     time.sleep(100)
        #     self.timestatus = False
        #     #     print('aaa')
        #     for i in range(10):
        #         print(i)
        #         self.camera_save = 1
        #         self.takePhoto()
        #         time.sleep(0.5)
        #     # self.camera_save = 1
        #     self.timestatus = True
        # print(self.camera_save)
# XY载物台
    def move_x(self):
        a=self.doubleSpinBox_10.value()
        self.g.GCommand(f'pPOSX={a}')
        self.g.GCommand('XQ  # MOVEX,1')
        # print('开始')
        self.xy_status()
        # print('停止')

    def move_y(self):
        a=self.doubleSpinBox_11.value()
        self.g.GCommand(f'pPOSY={a}')
        self.g.GCommand('XQ  # MOVEY,2')
        self.xy_status()
        # print('y停止')

    def reset_x(self):
        self.g.GCommand('XQ#HOMEX')

    def reset_y(self):
        self.g.GCommand('XQ#HOMEY')

    def is_non_zero(self,s):
        s = s.strip()
        if not s:
            return False
        try:
            num = float(s)
            return num != 0
        except ValueError:
            return False

    def xy_status(self):
        # print(self.g.GCommand('pMOVEX='))
        # print(self.g.GCommand('pMOVEY='))
        is_stopping = self.is_non_zero(self.g.GCommand('pMOVEX=')) or self.is_non_zero(self.g.GCommand('pMOVEY='))
        print(is_stopping)
        while is_stopping == 1:
            is_stopping = self.is_non_zero(self.g.GCommand('pMOVEX=')) or self.is_non_zero(self.g.GCommand('pMOVEY='))
            # print(is_stopping)
        time.sleep(0.1)

    def stagestatus_x(self):

        self.is_stopping_x = round(float(self.g.GCommand('pMOVEX=')))

        # print(round(float(self.is_stopping_x))==1)
        while self.is_stopping_x == 1:
            self.is_stopping_x = round(float(self.g.GCommand('pMOVEX=')))
            # print('x轴没停止',self.is_stopping_x)
        time.sleep(0.01)

    def stagestatus_y(self):
        # ss=self.g.GCommand('pMOVEX=')
        # print(ss)

        self.is_stopping_y = round(float(self.g.GCommand('pMOVEY=')))
        # print(self.is_stopping_z)
        while self.is_stopping_y == 1:
            self.is_stopping_y = round(float(self.g.GCommand('pMOVEY=')))
            # print('x轴没停止',self.is_stopping_z)
        time.sleep(0.01)
# XY位置
    def update_status(self):

        # print(self.doubleSpinBox_10.value())
        if self.timestatus:
            res_x = self.g.GCommand('TPA')
            res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            res_y = self.g.GCommand('TPB')
            res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            if res_x.strip():
                self.doubleSpinBox_100.setValue(round(float(res_x)) / 6400)
                self.pos_x = round(float(res_x)) / 6400
            if res_y.strip():
                self.doubleSpinBox_101.setValue(round(float(res_y)) / 6400)
                self.pos_y = round(float(res_y)) / 6400

            if res_x and res_y:
                self.pos_x = round(float(res_x)) / 6400
                self.pos_y = round(float(res_y)) / 6400

            self.takePhoto()

        # self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
        # pos = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
        # pos=self.Zmc.ZAux_Direct_GetEndMove(0)
        # Z轴位置


        end_position = self.get_z_pos()

        # self.doubleSpinBox_102.setValue(int(pos) / 40000)
        # self.doubleSpinBox_104.setValue(int(pos) / 40000)
        # print(type(pos))
        if end_position:
            self.doubleSpinBox_102.setValue(end_position)
            self.doubleSpinBox_104.setValue(end_position)

        # if len(self.z_pos_list) >= 3:
        #     self.z_pos_list = self.z_pos_list[1:]
        # self.z_pos_list.append(pos)
        #
        #
        #
        # if all(x == self.z_pos_list[0] for x in self.z_pos_list) and len(self.z_pos_list)>=3:
        #     self.is_stopping_z = True





        now = datetime.now()
        self.date_folder = now.strftime("%Y-%m-%d")  # 日期文件夹名称，例如 "2023-10-05"
        self.time_folder = now.strftime("%H-%M-%S")  # 时间文件夹名称，例如 "14-30-00"



        # if self.is_press:
        #     Zmc.ZAux_Direct_SetUserVar('relative_step', 2000)
        #     Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)

        # print(pos)

#     def img(self):
#         while True:
#             frame,number = self.queue.get()
#             if
#                 break
#     #
    def camera_get_img(self,frame,number):
# 相机按钮取图
#         print('chufa',self.timestatus,self.autofocus_flag)
#         if self.flag == 'zdjj':
#             self.queue.put((frame,number))

        self.get_image_flag_temp += 1
        # if self.get_image_flag_temp>=3:
        #     self.get_image_flag_temp = 0
        if self.camera_save == 1:
            if number == 'DA3827093':
                # print(number)
                # cv2.imwrite(f'frame_{self.camera_save}.jpg', frame)
                now = datetime.now()
                # date_folder = now.strftime("%Y-%m-%d")
                # time_folder = now.strftime("%H-%M-%S")
                # Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦面\{date_folder+"-"+time_folder}.bmp")
                if self.status == 'trigger':
                    img = Image.fromarray(frame)
                    # # 水平翻转（左右镜像）
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_img.save(
                         rf"{self.save_path_triggerScan}\焦面\{self.date_folder + '-' + self.time_folder}.bmp")
                    # Image.fromarray(frame).save(rf"{self.save_path_triggerScan}\焦面\{self.date_folder + "-" + self.time_folder}.bmp")
                # elif self.status == 'z_scan':
                #     Image.fromarray(frame).save(
                #         rf"{self.full_path}\{self.pos_z}.bmp")
                elif self.status == 'snake_scan':
                    img = Image.fromarray(frame)
                    # # 水平翻转（左右镜像）
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_img.save(
                        rf"{self.save_path_snakeScan}\焦面\{self.pos_x}-{self.pos_y}.bmp")
                    # Image.fromarray(frame).save(
                    #     rf"{self.save_path_snakeScan}\焦面\{self.pos_x}-{self.pos_y}.bmp")

                # cv2.imwrite(rf"E:\ftkpic\trigger\焦面\{timestamp}.jpg", frame)
                self.camera_save = 0
        if self.camera_save == 2:
            if number == 'DA3562117':
                # print(number)
                now = datetime.now()
                date_folder = now.strftime("%Y-%m-%d")
                time_folder = now.strftime("%H-%M-%S")
                if self.status == 'trigger':
                    Image.fromarray(frame).rotate(180).save(

                        rf"{self.save_path_triggerScan}\焦前\{self.date_folder + "-" + self.time_folder}.bmp")
                # elif self.status == 'z_scan':
                #     Image.fromarray(frame).save(
                #         rf"{self.full_path}\{self.pos_z}.bmp")
                elif self.status == 'snake_scan':
                    Image.fromarray(frame).save(
                    rf"{self.save_path_snakeScan}\焦前\{self.pos_x}-{self.pos_y}.bmp")

                self.camera_save = 0
        if self.camera_save == 3:
            if number == 'DA3562103':
                # print(number)
                now = datetime.now()
                date_folder = now.strftime("%Y-%m-%d")
                time_folder = now.strftime("%H-%M-%S")
                if self.status == 'trigger':
                    Image.fromarray(frame).save(
                        rf"{self.save_path_triggerScan}\焦后\{self.date_folder + "-" + self.time_folder}.bmp")
                # elif self.status == 'z_scan':
                #     Image.fromarray(frame).save(
                #         rf"{self.full_path}\{self.pos_z}.bmp")
                elif self.status == 'snake_scan':
                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦后\{self.pos_x}-{self.pos_y}.bmp")

                self.camera_save = 0
                self.status = None
        if self.camera_save == 4:
            self.camera_save_index += 1
            if self.status == 'snake_scan':

                print(number,frame.shape)
                img = frame
                if number == 'DA3562103':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\behind\{self.pos_x},{self.pos_y}.bmp",frame)
                    pic_path = rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp"
                    Image.fromarray(img).save(pic_path)
                if number == 'DA3562117':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\front\{self.pos_x},{self.pos_y}.bmp", frame)
                    pic_path = rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp"
                    Image.fromarray(img).save(pic_path)
                if number == 'DA3827093':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\middle\{self.pos_x},{self.pos_y}.bmp", frame)
                    pic_path = rf"{self.save_path_snakeScan}\焦面\{self.pos_x}-{self.pos_y}.bmp"
                    img = Image.fromarray(img)
                    # # 水平翻转（左右镜像）
                    # img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img.save(pic_path)
                    img = np.array(img)

                self.image_data[number] = {
                    'frame': img,
                    'pic_path': pic_path,
                    # 'result_pic_path':result_pic_path
                }
                if self.camera_save_index>=3:
                    if self.debug_image_queue:
                        result_pic_path = rf"{self.save_path_snakeScan}\检测结果\{self.pos_x},{self.pos_y}.jpg"
                        self.image_data['result_pic_path'] = result_pic_path
                        self.debug_image_queue.put(self.image_data)
                        self.image_data = {}
                    self.camera_save_index=0
                    self.camera_save = 0
                    self.status = None
        # print(self.timestatus,self.autofocus_flag)
        # pr

        if self.dp_focus_flag:
            self.dp_focus_img_list[f'{number}'] = frame

        if self.img_flag == 'layerscan':
            self.layerscan_queue.put((number,frame))


        if self.auto_focus_sample:

            self.list_pic.append(number)

            if number == 'DA3827093':
                # if self.camera_save == 5:
                if self.status == 'take_sample_pic':
                    path = rf"{self.cam1_path}\{self.auto_image_name[self.index_auto_focus_sample]}.jpg"
                    Image.fromarray(frame).save(path)
                self.camera_save = 0
            elif number == 'DA3562117':
                # if self.camera_save == 5:
                if self.status == 'take_sample_pic':
                    path = rf"{self.cam2_path}\{self.auto_image_name[self.index_auto_focus_sample]}.jpg"
                    Image.fromarray(frame).save(path)
                self.camera_save = 0
            elif number == 'DA3562103':
                # if self.camera_save == 5:
                if self.status == 'take_sample_pic':
                    path = rf"{self.cam3_path}\{self.auto_image_name[self.index_auto_focus_sample]}.jpg"
                    Image.fromarray(frame).save(path)
                self.camera_save = 0
            if len(self.list_pic)>=3:
                self.index_auto_focus_sample += 1
                self.list_pic = []
                if self.index_auto_focus_sample>100:
                    self.index_auto_focus_sample = 0
        # if self.camera_save == 5:
        #     print(number)
            # if self.status == 'take_sample_pic':
                # print(number)
                # if number == 'DA3827093':
                #     Image.fromarray(frame).save(
                #         rf"{self.cam1_path}\{self.pos_z - self.focus_pos}.bmp")
                # if number == 'DA3562117':
                #     Image.fromarray(frame).save(
                #         rf"{self.cam2_path}\{self.pos_z - self.focus_pos}.bmp")
                # if number == 'DA3562103':
                #
                #     Image.fromarray(frame).save(
                #         rf"{self.cam3_path}\{self.pos_z - self.focus_pos}.bmp")
            self.camera_save = 0

        if not self.timestatus:
            # print('计时器状态更新停止')
            # print(self.autofocus_flag)
            if self.autofocus_flag:
                # if number == 'DA3827093':
                if not self.flag_z_is_bottom:
                    pos_z = self.get_z_pos()
                    print('z轴到达',pos_z)
                    # print(self.Zmc.ZAux_Direct_GetIfIdle(0)[1].value)
                    self.z_pos_list.append(pos_z)
                    self.auto_frame_dict[f'{number}'] = frame
                    #     {
                    #         f'{number}':frame,
                    #     }
                    # )
                    if len(self.auto_frame_dict)>=3:
                        self.photo_number += 1
                        print('拍到',self.photo_number)
                        # print(self.flag_z_is_bottom)
                        self.autofocus_queue.put(self.auto_frame_dict)
                        self.auto_frame_dict = {}
                        if self.Zmc.ZAux_Direct_GetIfIdle(0)[1].value==-1:
                            print('tingzhi')
                            self.flag_z_is_bottom = True
                            self.photo_number = 0
                            self.autofocus_queue.put(self.flag_z_is_bottom)

            # if self.status == 'take_sample_pic':
            #     print(number)
            #     if number == 'DA3827093':
            #
            #         Image.fromarray(frame).save(
            #             rf"{self.cam1_path}\{self.pos_z - self.focus_pos}.bmp")
            #     if number == 'DA3562117':
            #
            #         Image.fromarray(frame).save(
            #             rf"{self.cam2_path}\{self.pos_z- self.focus_pos}.bmp")
            #     if number == 'DA3562103':
            #
            #         Image.fromarray(frame).save(
            #             rf"{self.cam3_path}\{self.pos_z - self.focus_pos}.bmp")
# 三个相机显示
        if len(frame.shape) == 3:
            print(number)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                frame.data, width, height, bytes_per_line,
                QImage.Format_RGB888
            ).rgbSwapped()  # BGR->RGB
            label = self.label_13  # 相机1->label_9，相机2->label_7
            pixmap = QPixmap.fromImage(q_img)
            scaled_pix = pixmap.scaled(
                label.width(), label.height(),
                # Qt.KeepAspectRatio,
                # Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pix)
        else:

            height, width = frame.shape
            q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)

        if number == 'DA3827093':
            transform = QTransform().scale(-1, 1)  # 水平翻转
            # 使用 transform 对 q_img 进行转换，并生成新的 QImage
            mirrored_img = q_img.transformed(transform)
            # 将 QImage 转换为 QPixmap 并缩放尺寸，然后设置到 label_9 上
            # self.label_9.setPixmap(QPixmap.fromImage(mirrored_img).scaled(width // 2, height // 2))
            # self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
            self.label_9.setPixmap(QPixmap.fromImage(mirrored_img).scaled(width // 4, height // 4))


        if number == 'DA3562117':
           transform = QTransform().rotate(180)  # Rotate 180 degrees
           rotated_img = q_img.transformed(transform)
           self.label_10.setPixmap(QPixmap.fromImage(rotated_img).scaled(width // 4, height // 4))
        if number == 'DA3562103':
           self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
           # cv2.imwrite('frame.jpg',frame)
        # if number == 1:
        #     print(number)
        #     self.label_13.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
        if number == 'DA3827093':
            # print(number)
            self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))

        # if number == 'DA6630523':
        #     self.label_13.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
        #     try:
                # 使用analyze_image方法直接分析图像
                # area, intensity, x_sol_area, x_sol_intensity = self.analyzer.analyze_image(frame,use_roi=False,visualize=False) # 不显示可视化窗口，只返回结果)
                # area, intensity, x_sol_area, x_sol_intensity
                #result_text = f"面积: {area:.2f} mm²\n强度: {intensity:.2f}\nx_area: {x_sol_area:.4f}\nx_intensity: {x_sol_intensity:.4f}"
            #     result_text = (
            #         f"面积(px): {area:.2f}\n"
            #         f"强度: {intensity:.2f}\n"
            #         f"x_area(laser): {self.laser_x_est if self.laser_x_est is not None else 'None'}"
            #     )
            #     self.label_14.setText(result_text)
            #
            #     self.label_14.setText(result_text)
            #     self.last_result_cam3 = (area, intensity, x_sol_area, x_sol_intensity)
            # except Exception as e:
            #             # 分析失败不影响显示
            #         print(f"[相机分析失败] {e}")
            # self.laser_area_last = area
            # x_ref = self.laser_x_est if self.laser_x_est is not None else None
            # x_est, valid = self.estimate_x_from_area_laser(area, x_ref)
            # self.laser_x_est = x_est
            # self.laser_x_valid = valid
        if number == 'DA6630523':
            self.label_13.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
            result = self.get_laser_spot_center(frame)
            if result is not None:
                cx, cy, area, z_position, is_before_focus = result
                # 更新状态
                self.last_laser_center = (cx, cy)
                self.last_laser_area = area
                self.current_z_position_mm = z_position
                self.is_before_focus = is_before_focus
                # 计算与焦点的距离
                focus_z = self.laser_focus_z_position
                delta_z_mm = z_position - focus_z
                delta_z_um = delta_z_mm * 1000
                # 显示详细信息
                focus_status = "焦前" if is_before_focus else "焦后"
                info_text = (
                    f"当前Z: {z_position:.3f} mm\n"
                    f"焦点Z: {focus_z:.3f} mm\n"
                    f"离焦: {delta_z_um:.1f} μm"
                )
                self.label_14.setText(info_text)  # 如果没有label_14，改成你的显示控件
            else:
                self.last_laser_center = None
                self.last_laser_area = None
                self.current_z_position_mm = None
                self.is_before_focus = None
                self.label_14.setText("未检测到光斑")
# 打印结果到控制台
    #         print(f"DA3827093相机分析结果:")
    #         print(f"  光斑面积: {area:.4f} mm²")
    #         print(f"  光斑强度: {intensity:.4f}")
    #         print(f"  面积对应x值: {x_sol_area:.6f}")
    #         print(f"  强度对应x值: {x_sol_intensity:.6f}")
    #
    # def camera_get_img(self,frame,number):
    #     if self.img_flag == 'layerscan':
    #         self.layerscan_queue.put((number,frame))

    def receive_img(self):
        index = 0
        scan_up = self.doubleSpinBox_9.value()
        scan_down = self.doubleSpinBox_14.value()
        scan_step = self.doubleSpinBox_13.value()
        while True:
            try:
                item = self.layerscan_queue.get(timeout=0.1)
                if item[0]=='DA3827093':#第一个相机采图
                #if item[0] == 'DA6630523':#第四个相机采图
                    # index += 1
                    # print(index)
                    dis=round(scan_up-scan_step*index,4)
                    frame = item[1]
                    Image.fromarray(frame).save(rf"{self.full_path}\{dis}.bmp")
                    index += 1
            except queue.Empty:
                if self.img_flag != 'layerscan':
                    break


    def timestatus_mode(self):
        self.timestatus = not self.timestatus
        self.auto_focus_sample = False

    def closeEvent(self, event):
        """
        重写 closeEvent 方法，隐藏窗口而非关闭
        """
        event.ignore()  # 忽略默认的关闭事件
        self.hide()  # 隐藏窗口
        self.is_showing=False



# 主程序初始化（测试）
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = Debug_UI(mode ='test')
    sys.exit(app.exec_())


