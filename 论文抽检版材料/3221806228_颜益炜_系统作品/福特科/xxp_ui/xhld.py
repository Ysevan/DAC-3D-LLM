import ctypes
import shutil
import threading
import time
from ctypes import memset, cast, c_ubyte
import os
from datetime import datetime
from time import sleep
import queue
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, uic
import sys
import gclib
from PyQt5.QtCore import QTimer, QDateTime, QEvent
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from _ctypes import byref, sizeof, POINTER

from window.control.MvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX, MV_SAVE_IMAGE_TO_FILE_PARAM_EX, \
    MV_Image_Bmp
from window.control.camera import CameraFactory
from zmcdll.zauxdllPython import ZAUXDLL
from window.control.light import lightControllor
from window.control.getposxy import getposxy

class Debug_UI(QtWidgets.QMainWindow):
    timer = QTimer()

    def __init__(self,debug_ui_queue=None,ui_debug_queue=None,debug_image_queue=None,image_debug_queue=None,image_queue = None,mode='mainUICall'):
        super(Debug_UI, self).__init__()

        self.pos_z = None
        self.debug_image_queue = debug_image_queue

        self.image_debug_queue = image_debug_queue
        self.debug_ui_queue = debug_ui_queue
        self.ui_debug_queue = ui_debug_queue

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

        self.g = gclib.py()
        self.cam = CameraFactory()
        self.Zmc = ZAUXDLL()
        self.light = lightControllor()

        print('gclib version:', self.g.GVersion())
        self.g.GOpen('10.0.0.100 --direct -s ALL')
        strtemp = '192.168.0.11'

        iresult = self.Zmc.ZAux_OpenEth(strtemp)
        if 0 != iresult:
            print('连接失败')
        else:
            print('连接成功')

        self.timestatus=True
        self.xy_pos = []
        self.xy_pos = getposxy(-10.7,-27.1)
        self.get_image_flag_temp = 0


        self.is_stopping_z = 0
        self.z_pos_list = []
        self.x_pos_list=[]
        self.save_path_layerScan = r'E:\ftkpic\layerscan'
        self.save_path_snakeScan = r'E:\ftkpic\snakescan'
        self.save_path_triggerScan = r'E:\ftkpic\trigger'

        camList = self.cam.setFunc(self.camera_get_img)# 得到三相机cam句柄
        custom_order = {"DA3827093": 0, "DA3562117": 1, "DA3562103": 2,"DA6630523": 3}

        # 使用 sorted 函数，并使用 lambda 表达式作为 key 参数
        self.camList = sorted(camList, key=lambda camList: custom_order[camList['number']])
        self.step_z = 0

        self.camera_save = 0
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

        # self.btn_pushButton_12.clicked.connect(self.zcan)
        self.spinBox.valueChanged.connect(self.onValueChanged_cam1)
        self.spinBox_3.valueChanged.connect(self.onValueChanged_cam2)
        self.spinBox_2.valueChanged.connect(self.onValueChanged_cam3)
        self.spinBox_15.valueChanged.connect(self.onValueChanged_Zspeed)

        self.pushButton_24.clicked.connect(self.scan_xy)



        # self.btn_save_73.clicked.connect(self.move_z_up)

        self.pushButton_19.clicked.connect(self.save_pic)
        self.pushButton_18.clicked.connect(self.save_pic)
        self.pushButton_20.clicked.connect(self.save_pic)

        self.btn_save_73.installEventFilter(self)
        self.btn_save_72.installEventFilter(self)
        self.btn_save_81.installEventFilter(self)
        self.btn_save_80.installEventFilter(self)
        self.pushButton_12.clicked.connect(self.zscan)



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

        self.btn_save_76.clicked.connect(
            lambda: self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 4)
        )

        self.timer.timeout.connect(self.update_status)
        self.timer.start(100)

        self.getInfoFromUI_thread = threading.Thread(target=self.getInfoFromUI)
        self.getInfoFromUI_thread.start()

    def getInfoFromUI(self):
        if self.ui_debug_queue:
            while True:
                try:
                    msg = self.ui_debug_queue.get()
                    if msg:
                        print(msg)
                        if msg['type'] == 'Control':
                            if msg['func'] == 'Scan':
                                self.scan_xy()
                                # self.timestatus = False
                                # self.takePhoto()

                            if msg['func'] == 'Stop':
                                self.stop_flag = True
                            if msg['func'] == 'Move':
                                button_num = int(msg['button_num'])
                                self.g.GCommand(f'pPOSX={ self.xy_pos[button_num][0]}')
                                self.g.GCommand('XQ  # MOVEX,1')
                                self.g.GCommand(f'pPOSY={ self.xy_pos[button_num][1]}')
                                self.g.GCommand('XQ  # MOVEY,2')
                except queue.Empty:
                    pass

    def eventFilter(self, obj, event):
        if obj == self.btn_save_73 or obj == self.btn_save_81:
            if event.type() == QEvent.MouseButtonPress:
                self.is_press_up = True
                thread = threading.Thread(target=self.move_z_relative_up)
                thread.start()
            elif event.type() == QEvent.MouseButtonRelease:
                self.is_press_up = False
        if obj == self.btn_save_72 or obj == self.btn_save_80:
            if event.type() == QEvent.MouseButtonPress:
                self.is_press_down = True
                thread = threading.Thread(target=self.move_z_relative_down)
                thread.start()
            elif event.type() == QEvent.MouseButtonRelease:
                self.is_press_down = False
        return super(Debug_UI, self).eventFilter(obj, event)

# 蛇形扫描
    def scan_xy(self):
        delaytime=1
        self.save_path_snakeScan += '//'+self.date_folder + '-' + self.time_folder
        os.makedirs(self.save_path_snakeScan, exist_ok=True)
        os.makedirs(self.save_path_snakeScan+'/焦前', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦后', exist_ok=True)
        os.makedirs(self.save_path_snakeScan + '/焦面', exist_ok=True)

        x_pos_start = self.doubleSpinBox_16.value()
        y_pos_start = self.doubleSpinBox_17.value()
        self.xy_pos = []
        self.xy_pos = getposxy(x_pos_start,y_pos_start)

        thread_xy_scan = threading.Thread(target=self.xy_scan_thread, args=(self.xy_pos, ))
        thread_xy_scan.start()

    def xy_scan_thread(self, xy_pos):
        try:
            start_time = time.time()
            self.status = 'xy_scan'

            # print('xy_scan_thread')
            # l=1

            # self.g.GCommand(f'pPOSX={x_pos_start}')
            self.g.GCommand(f'pPOSX={self.xy_pos[0][0]}')
            self.g.GCommand('XQ  # MOVEX,1')

            self.g.GCommand(f'pPOSY={self.xy_pos[0][1]}')
            # self.g.GCommand(f'pPOSY={y_pos_start}')

            self.g.GCommand('XQ  # MOVEY,2')
            time.sleep(2)
            self.timestatus = False
            time.sleep(3)
            self.get_image_flag_temp = 0
            res_x = self.g.GCommand('TPA')
            res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            res_y = self.g.GCommand('TPB')
            res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            if res_x.strip():
                self.doubleSpinBox_100.setValue(int(res_x) / 6400)
                self.pos_x = int(res_x) / 6400
            if res_y.strip():
                self.doubleSpinBox_101.setValue(int(res_y) / 6400)
                self.pos_y = int(res_y) / 6400

            self.autofocus()
            while self.autofocus_flag:  # 等待自动对焦完成
                pass
            self.get_image_flag_temp = 0
            self.pos_x = int(res_x) / 6400
            self.pos_y = int(res_y) / 6400

            self.status = 'snake_scan'
            self.camera_save = 4
            self.takePhoto()

            while self.get_image_flag_temp<3:
                pass
            self.get_image_flag_temp = 0



            for j in range(11):
                    if self.stop_flag:
                        break
                    for i in range(11):
                        if self.stop_flag:
                            break

                        print(xy_pos[i + j * 12 + 1][0])
                        self.g.GCommand(f'pPOSX={self.xy_pos[i+j*12+1][0]}')
                        self.g.GCommand('XQ  # MOVEX,1')
                        time.sleep(0.5)
                        res_x = self.g.GCommand('TPA')
                        res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                        res_y = self.g.GCommand('TPB')
                        res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                        if res_x.strip():
                            self.doubleSpinBox_100.setValue(int(res_x) / 6400)
                            self.pos_x = int(res_x) / 6400
                        if res_y.strip():
                            self.doubleSpinBox_101.setValue(int(res_y) / 6400)
                            self.pos_y = int(res_y) / 6400
                        if self.pos_x and self.pos_y:
                            self.pos_x = int(res_x) / 6400
                            self.pos_y = int(res_y) / 6400
                        self.autofocus()
                        while self.autofocus_flag:  # 等待自动对焦完成
                            pass

                        self.status = 'snake_scan'
                        self.camera_save = 4
                        self.takePhoto()
                        while self.get_image_flag_temp < 3:
                            pass
                        self.get_image_flag_temp = 0


                        # print(xy_pos[i][0])
                    print(xy_pos[j*12+13][1])
                    self.g.GCommand(f'pPOSY={self.xy_pos[j*12+13][1]}')
                    self.g.GCommand('XQ  # MOVEY,2')
                    time.sleep(0.5)
                    res_x = self.g.GCommand('TPA')
                    res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    res_y = self.g.GCommand('TPB')
                    res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                    if res_x.strip():
                        self.doubleSpinBox_100.setValue(int(res_x) / 6400)
                        self.pos_x = int(res_x) / 6400
                    if res_y.strip():
                        self.doubleSpinBox_101.setValue(int(res_y) / 6400)
                        self.pos_y = int(res_y) / 6400

                    self.pos_x = int(res_x) / 6400
                    self.pos_y = int(res_y) / 6400
                    self.autofocus()
                    while self.autofocus_flag:  # 等待自动对焦完成
                        pass
                    self.camera_save = 4
                    self.status = 'snake_scan'
                    self.takePhoto()
                    while self.get_image_flag_temp < 3:
                        pass
                    self.get_image_flag_temp = 0
            for k in range(11):

                if self.stop_flag:
                    break
                # print(xy_pos[133+k][0])
                self.g.GCommand(f'pPOSX={self.xy_pos[133+k][0]}')
                self.g.GCommand('XQ  # MOVEX,1')
                time.sleep(0.5)
                res_x = self.g.GCommand('TPA')
                res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                res_y = self.g.GCommand('TPB')
                res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
                if res_x.strip():
                    self.doubleSpinBox_100.setValue(int(res_x) / 6400)
                    self.pos_x = int(res_x) / 6400
                if res_y.strip():
                    self.doubleSpinBox_101.setValue(int(res_y) / 6400)
                    self.pos_y = int(res_y) / 6400

                self.pos_x = int(res_x) / 6400
                self.pos_y = int(res_y) / 6400
                self.autofocus()
                while self.autofocus_flag:  # 等待自动对焦完成
                    pass

                self.status = 'snake_scan'
                self.camera_save = 4
                self.takePhoto()

                while self.get_image_flag_temp < 3:
                    pass
                self.get_image_flag_temp = 0
            self.timestatus = True
            end_time = time.time()
            print(f"蛇形扫描用时: {end_time - start_time} seconds")
        except Exception as e:
            print(e)

    def autofocus(self):
        self.autofocus_flag = True

        self.takePhoto()
        thread = threading.Thread(target=self.getAutoFocusImg)
        thread.start()

    def getAutoFocusImg(self):

        while True:
            try:
                item = self.autofocus_queue.get(timeout=2)
                if isinstance(item,np.ndarray):
                    print(f'自动对焦图像：高{item.shape[0]},宽{item.shape[1]}')
                    sleep(0.5)
                    # 自动对焦逻辑


                    break

            except queue.Empty:
                pass
        self.autofocus_flag = False
        self.get_image_flag_temp = 0


#Z轴控制
    def move_z_relative_up(self):
        while True:
            if self.is_press_up:

                self.Zmc.ZAux_Direct_SetUserVar('relative_step', 1000)
                self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
                sleep(0.05)
            else:
                break
    def move_z_relative_down(self):

        while True:
            if self.is_press_down:

                self.Zmc.ZAux_Direct_SetUserVar('relative_step', -1000)
                self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
                sleep(0.05)
            else:
                break
    def stop_z(self):
        self.Zmc.ZAux_Direct_SetUserVar('relative_step', 0)
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)


# Z轴绝对运动
    def move_z_up(self,pos):
        pos=pos*40000
        self.Zmc.ZAux_Direct_SetUserVar('absolute_step', pos)
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 2)

    def onValueChanged_Zspeed(self, val):
        speed = val * 600 / 1000
        self.Zmc.ZAux_Direct_SetUserVar('z_speed', speed)
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 5)

    def move_z(self):
        # print("2222")
        a = self.doubleSpinBox_12.value()
        self.move_z_up(a)
#    层扫
    def zscan(self):

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

        thread_z_scan = threading.Thread(target=self.z_scan_thread,args=(scan_up,scan_down,scan_step,))
        thread_z_scan.start()

    def z_status(self):
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 6)
        self.is_stopping_z = self.Zmc.ZAux_Direct_GetUserVar('is_stopping')[1].value
        while self.is_stopping_z:
            self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 6)
            self.is_stopping_z = self.Zmc.ZAux_Direct_GetUserVar('is_stopping')[1].value
            print('z轴没停止',self.is_stopping_z)
            continue

    def z_scan_thread(self,scan_up,scan_down,scan_step):
        self.timestatus = False
        self.status = 'z_scan'
        self.Zmc.ZAux_Direct_SetUserVar('z_speed', 500)
        # self.Zmc.ZAux_Direct_SetUserVar('z_speed', (scan_up-scan_down)*100/5)
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 5)
        # print('z_scan_thread')
        sleep(0.5)
        self.move_z_up(scan_up)
        # self.z_status()
        sleep(2)
        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
        self.pos_z = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
        print(self.pos_z/40000)
        self.camera_save = 1
        self.status = 'z_scan'
        self.takePhoto()
        while self.pos_z/40000>scan_down:
            self.move_z_up(self.pos_z/40000-scan_step)
            # self.z_status()
            sleep(1)
            self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
            self.pos_z = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
            print(self.pos_z)
            self.camera_save = 3
            self.status = 'z_scan'
            self.takePhoto()

        self.timestatus = True

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
            if os.path.isfile(item_path) or os.path.islink(item_path):
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

    def takePhoto(self):

        self.Zmc.ZAux_Direct_SetOp(0, 1)
        self.Zmc.ZAux_Direct_SetOp(1, 1)
        self.Zmc.ZAux_Direct_SetOp(2, 1)
        sleep(0.01)
        self.Zmc.ZAux_Direct_SetOp(0, 0)
        self.Zmc.ZAux_Direct_SetOp(1, 0)
        self.Zmc.ZAux_Direct_SetOp(2, 0)
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
        print(self.camera_save)

# XY载物台
    def move_x(self):
        print('')
        a=self.doubleSpinBox_10.value()
        self.g.GCommand(f'pPOSX={a}')
        self.g.GCommand('XQ  # MOVEX,1')
    def move_y(self):
        print('')
        a=self.doubleSpinBox_11.value()
        self.g.GCommand(f'pPOSY={a}')
        self.g.GCommand('XQ  # MOVEY,2')


    def reset_x(self):
        self.g.GCommand('XQ#HOMEX')

    def reset_y(self):
        self.g.GCommand('XQ#HOMEY')
# XY位置
    def update_status(self):

        # print(self.doubleSpinBox_10.value())
        if self.timestatus:
            res_x = self.g.GCommand('TPA')
            res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            res_y = self.g.GCommand('TPB')
            res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
            if res_x.strip():
                self.doubleSpinBox_100.setValue(int(res_x) / 6400)
                self.pos_x = int(res_x) / 6400
            if res_y.strip():
                self.doubleSpinBox_101.setValue(int(res_y) / 6400)
                self.pos_y = int(res_y) / 6400

            if res_x and res_y:
                self.pos_x = int(res_x) / 6400
                self.pos_y = int(res_y) / 6400

            self.takePhoto()

        self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
        pos = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
        self.pos=pos
        self.doubleSpinBox_102.setValue(int(pos) / 40000)
        self.doubleSpinBox_104.setValue(int(pos) / 40000)

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
    def camera_get_img(self,frame,number):
# 相机按钮取图
#         print(number)
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
                    Image.fromarray(frame).save(rf"{self.save_path_triggerScan}\焦面\{self.date_folder + "-" + self.time_folder}.bmp")
                elif self.status == 'z_scan':
                    Image.fromarray(frame).save(
                        rf"{self.full_path}\{self.pos_z/40000}.bmp")
                elif self.status == 'snake_scan':
                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦面\{self.pos_x}-{self.pos_y}.bmp")

                # cv2.imwrite(rf"E:\ftkpic\trigger\焦面\{timestamp}.jpg", frame)
                self.camera_save = 0
        if self.camera_save == 2:
            if number == 'DA3562117':
                # print(number)
                now = datetime.now()
                date_folder = now.strftime("%Y-%m-%d")
                time_folder = now.strftime("%H-%M-%S")
                if self.status == 'trigger':
                    Image.fromarray(frame).save(
                        rf"{self.save_path_triggerScan}\焦前\{self.date_folder + "-" + self.time_folder}.bmp")
                elif self.status == 'z_scan':
                    Image.fromarray(frame).save(
                        rf"{self.full_path}\{self.pos_z/40000}.bmp")
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
                elif self.status == 'z_scan':
                    Image.fromarray(frame).save(
                        rf"{self.full_path}\{self.pos_z/40000}.bmp")
                elif self.status == 'snake_scan':
                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦后\{self.pos_x}-{self.pos_y}.bmp")

                self.camera_save = 0
                self.status = None
        if self.camera_save == 4:
            self.camera_save_index += 1
            if self.status == 'snake_scan':

                print(number,frame.shape)

                if number == 'DA3562103':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\behind\{self.pos_x},{self.pos_y}.bmp",frame)

                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp")
                if number == 'DA3562117':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\front\{self.pos_x},{self.pos_y}.bmp", frame)
                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp")
                if number == 'DA3827093':
                    # cv2.imwrite(rf"{self.save_path_snakeScan}\middle\{self.pos_x},{self.pos_y}.bmp", frame)
                    Image.fromarray(frame).save(
                        rf"{self.save_path_snakeScan}\焦面\{self.pos_x},{self.pos_y}.bmp")
                if self.camera_save_index>=3:
                    self.camera_save_index=0
                    self.camera_save = 0
                    self.status = None

        if not self.timestatus:
            # print('计时器状态更新停止')
            if self.autofocus_flag:
                if number == 'DA3827093':
                    self.autofocus_queue.put(frame)#传递焦面图片到自动对焦函数
            else:
                # self.image_data[number]=frame
                if number == 'DA3562103':
                    pic_path = rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp"
                if number == 'DA3562117':
                    pic_path = rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp"
                if number == 'DA3827093':
                    pic_path = rf"{self.save_path_snakeScan}\焦面\{self.pos_x},{self.pos_y}.bmp"

                self.image_data[number] = {
                    'frame':frame,
                    'pic_path':pic_path
                }
                if len(self.image_data)>=3:
                    # print('图片传到图像处理进程')
                    #
                    if self.debug_image_queue:
                        self.debug_image_queue.put(self.image_data)
                        self.image_data = {}
# 三个相机显示
        height, width = frame.shape
        q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        if number == 'DA3827093':
            self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
        if number == 'DA3562117':
           self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
        if number == 'DA3562103':
           self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
                # cv2.imwrite('frame.jpg',frame)
    # def camera_get_img(self, cam, pData=0, nDataSize=0,number=0):  # 相机取图线程
    #     stFrameInfo = MV_FRAME_OUT_INFO_EX()
    #     memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    #     while True:
    #         # print('wait')
    #         try:
    #             ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 10000)
    #             if ret == 0:
    #                 # print("get one frame: Width[%d], Height[%d], nFrameNum[%d], number[%s]" % (
    #                 #     stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum,number))
    #
    #                 frame = np.array(pData)  # 将c_ubyte_Array转化成ndarray得到（5308416，）
    #                 frame = cv2.resize(frame, (2048, 2048))
    #                 height, width = frame.shape
    #                 q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
    #                 # self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 # self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 # self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 # 相机按钮取图
    #                 if self.camera_save == 1:
    #                     if number == 'DA3827093':
    #                         print(number)
    #                         # cv2.imwrite(f'frame_{self.camera_save}.jpg', frame)
    #                         now = datetime.now()
    #                         date_folder = now.strftime("%Y-%m-%d")
    #                         time_folder = now.strftime("%H-%M-%S")
    #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦面\{date_folder+"-"+time_folder}.bmp")
    #                         # cv2.imwrite(rf"E:\ftkpic\trigger\焦面\{timestamp}.jpg", frame)
    #                         self.camera_save = 0
    #                 if self.camera_save == 2:
    #                     if number == 'DA3562117':
    #                         print(number)
    #                         now = datetime.now()
    #                         date_folder = now.strftime("%Y-%m-%d")
    #                         time_folder = now.strftime("%H-%M-%S")
    #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦前\{date_folder+"-"+time_folder}.bmp")
    #                         self.camera_save = 0
    #                 if self.camera_save == 3:
    #                     if number == 'DA3562103':
    #                         print(number)
    #                         now = datetime.now()
    #                         date_folder = now.strftime("%Y-%m-%d")
    #                         time_folder = now.strftime("%H-%M-%S")
    #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦后\{date_folder+"-"+time_folder}.bmp")
    #                         self.camera_save = 0
    #                 # 三个相机显示
    #                 if number == 'DA3827093':
    #                     self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 if number == 'DA3562117':
    #                    self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 if number == 'DA3562103':
    #                    self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #                 # cv2.imwrite('frame.jpg',frame)
    #
    #                 # cv2.imwrite("D:/123456/frame%d.jpg" % stFrameInfo.nFrameNum, frame)
    #                 # height, width = frame.shape
    #
    #                 # 将OpenCV灰度图像转换为QImage
    #                 # 单通道图像需要使用不同的QImage.Format
    #                 # if self.ui:
    #                 #     q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
    #                 #     self.ui.imageLabel.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
    #
    #                 # cv2.imshow("temp", temp)
    #                 # cv2.waitKey(0)
    #
    #         except Exception as e:
    #             print('相机报错:', e)




# 主程序初始化
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = Debug_UI(mode ='test')
    sys.exit(app.exec_())

#
# import ctypes
# import shutil
# import threading
# import time
# from ctypes import memset, cast, c_ubyte
# import os
# from datetime import datetime
# from time import sleep
# import queue
# import cv2
# import numpy as np
# from PIL import Image
# from PyQt5 import QtWidgets, uic
# import sys
# import gclib
# from PyQt5.QtCore import QTimer, QDateTime, QEvent
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QFileDialog, QMessageBox
# from _ctypes import byref, sizeof, POINTER
#
# # 导入瑕疵检测相关模块
# import torch
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# from ultralytics.utils.plotting import Annotator, colors
#
# # 导入其他必要模块
# from window.control.MvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX, MV_SAVE_IMAGE_TO_FILE_PARAM_EX, \
#     MV_Image_Bmp
# from window.control.camera import CameraFactory
# from zmcdll.zauxdllPython import ZAUXDLL
# from window.control.light import lightControllor
# from window.control.getposxy import getposxy
#
#
#
# class Debug_UI(QtWidgets.QMainWindow):
#     timer = QTimer()
#
#     def __init__(self, debug_ui_queue=None, ui_debug_queue=None, debug_image_queue=None, image_debug_queue=None,
#                  image_queue=None, mode='mainUICall'):
#         super(Debug_UI, self).__init__()
#
#         self.debug_image_queue = debug_image_queue
#
#         self.image_debug_queue = image_debug_queue
#         self.debug_ui_queue = debug_ui_queue
#         self.ui_debug_queue = ui_debug_queue
#
#         # 添加瑕疵检测队列-可供yolo算法使用
#         self.defect_detection_queue = queue.Queue()
#         self.defect_results_queue = queue.Queue()
#
#         self.autofocus_queue = queue.Queue() #这应该是自动对焦队列，使用了线程的方式
#         self.autofocus_flag = False
#         self.stop_flag = False
#         self.image_queue = image_queue
#         self.image_data = {}
#         self.pos_y = None
#         self.pos_x = None
#         self.full_path = None
#         self.status = None
#         self.pos = None
#         self.is_press_down = None
#         self.is_press_up = None
#         self.camera_save_index = 0
#
#
#         self.g = gclib.py()
#         self.cam = CameraFactory()
#         self.Zmc = ZAUXDLL()
#         self.light = lightControllor()
#
#         print('gclib version:', self.g.GVersion())
#         self.g.GOpen('10.0.0.100 --direct -s ALL')
#         strtemp = '192.168.0.11'
#
#         iresult = self.Zmc.ZAux_OpenEth(strtemp)
#         if 0 != iresult:
#             print('连接失败')
#         else:
#             print('连接成功')
#
#         self.timestatus = True
#         self.xy_pos = []
#         self.xy_pos = getposxy(-10.7, -27.1)
#         self.get_image_flag_temp = 0
#
#         self.is_stopping_z = True
#         self.z_pos_list = []
#         self.x_pos_list = []
#
#         # 初始化瑕疵检测模型
#         self.defect_model = None
#         self.defect_model_type = None
#         self.defect_detection_enabled = False
#
#         # 添加瑕疵检测结果保存路径
#         self.save_path_defect_results = r'E:\ftkpic\defect_results'
#         os.makedirs(self.save_path_defect_results, exist_ok=True)#如果没有这个文件夹就创建一个文件及
#
#         self.save_path_layerScan = r'E:\ftkpic\layerscan'
#         self.save_path_snakeScan = r'E:\ftkpic\snakescan'
#         self.save_path_triggerScan = r'E:\ftkpic\trigger'
#
#         camList = self.cam.setFunc(self.camera_get_img)  # 得到三相机cam句柄
#         custom_order = {"DA3827093": 0, "DA3562117": 1, "DA3562103": 2}
#
#         # 使用 sorted 函数，并使用 lambda 表达式作为 key 参数
#         self.camList = sorted(camList, key=lambda camList: custom_order[camList['number']])
#         self.step_z = 0
#         self.camera_save = 0
#
#         # 启动瑕疵检测线程
#         self.defect_detection_thread = threading.Thread(target=self.defect_detection_worker)
#         self.defect_detection_thread.daemon = True
#         self.defect_detection_thread.start()
#
#         # 按钮绑定
#         uic.loadUi(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\ui\debug.ui', self)  # 加载UI文件
#         if not mode == 'mainUICall':
#             self.show()
#         self.btn_save_105.clicked.connect(self.move_x)
#         self.btn_save_106.clicked.connect(self.move_y)
#         self.btn_save_61.clicked.connect(self.reset_x)
#         self.btn_save_66.clicked.connect(self.reset_y)
#         self.btn_save_107.clicked.connect(self.move_z)
#         self.pushButton_21.clicked.connect(self.stop_z)
#         self.pushButton_13.clicked.connect(self.deletepic)
#
#         # self.btn_pushButton_12.clicked.connect(self.zcan)
#         self.spinBox.valueChanged.connect(self.onValueChanged_cam1)
#         self.spinBox_3.valueChanged.connect(self.onValueChanged_cam2)
#         self.spinBox_2.valueChanged.connect(self.onValueChanged_cam3)
#         self.spinBox_15.valueChanged.connect(self.onValueChanged_Zspeed)
#
#         self.pushButton_24.clicked.connect(self.scan_xy)
#
#         # self.btn_save_73.clicked.connect(self.move_z_up)
#
#         self.pushButton_19.clicked.connect(self.save_pic)
#         self.pushButton_18.clicked.connect(self.save_pic)
#         self.pushButton_20.clicked.connect(self.save_pic)
#
#         self.btn_save_73.installEventFilter(self)
#         self.btn_save_72.installEventFilter(self)
#         self.btn_save_81.installEventFilter(self)
#         self.btn_save_80.installEventFilter(self)
#         self.pushButton_12.clicked.connect(self.zscan)
#
#         self.pushButton.clicked.connect(
#             lambda: self.light.light15control(1)
#         )
#         self.pushButton_16.clicked.connect(
#             lambda: self.light.light15control(0)
#         )
#         self.pushButton_17.clicked.connect(
#             lambda: self.light.light60control(1)
#         )
#         self.pushButton_31.clicked.connect(
#             lambda: self.light.light60control(0)
#         )
#
#         self.btn_save_76.clicked.connect(
#             lambda: self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 4)
#         )
#
#         # 添加瑕疵检测开关按钮
#         self.pushButton_defect = QtWidgets.QPushButton(self)
#         self.pushButton_defect.setGeometry(450, 500, 120, 30)
#         self.pushButton_defect.setText("启用瑕疵检测")
#         self.pushButton_defect.clicked.connect(self.toggle_defect_detection)
#
#         self.timer.timeout.connect(self.update_status)
#         self.timer.start(100)
#
#         self.getInfoFromUI_thread = threading.Thread(target=self.getInfoFromUI)
#         self.getInfoFromUI_thread.start()
#
#         # 添加瑕疵检测开关方法
#
#
#     def getInfoFromUI(self):
#         if self.ui_debug_queue:
#             while True:
#                 try:
#                     msg = self.ui_debug_queue.get()
#                     if msg:
#                         print(msg)
#                         if msg['type'] == 'Control':
#                             if msg['func'] == 'Scan':
#                                 self.scan_xy()
#                                 # self.timestatus = False
#                                 # self.takePhoto()
#
#                             if msg['func'] == 'Stop':
#                                 self.stop_flag = True
#                             if msg['func'] == 'Move':
#                                 button_num = int(msg['button_num'])
#                                 self.g.GCommand(f'pPOSX={self.xy_pos[button_num][0]}')
#                                 self.g.GCommand('XQ  # MOVEX,1')
#                                 self.g.GCommand(f'pPOSY={self.xy_pos[button_num][1]}')
#                                 self.g.GCommand('XQ  # MOVEY,2')
#                 except queue.Empty:
#                     pass
#
#     def eventFilter(self, obj, event):
#         if obj == self.btn_save_73 or obj == self.btn_save_81:
#             if event.type() == QEvent.MouseButtonPress:
#                 self.is_press_up = True
#                 thread = threading.Thread(target=self.move_z_relative_up)
#                 thread.start()
#             elif event.type() == QEvent.MouseButtonRelease:
#                 self.is_press_up = False
#         if obj == self.btn_save_72 or obj == self.btn_save_80:
#             if event.type() == QEvent.MouseButtonPress:
#                 self.is_press_down = True
#                 thread = threading.Thread(target=self.move_z_relative_down)
#                 thread.start()
#             elif event.type() == QEvent.MouseButtonRelease:
#                 self.is_press_down = False
#         return super(Debug_UI, self).eventFilter(obj, event)
#
#     #yolo算法调用
#     def toggle_defect_detection(self):
#         if not self.defect_detection_enabled:
#             # 加载模型
#             try:
#                 model_path = r"d:\zycgit\ZDevelop_Confocal\xxp_ui\models\best.pt"
#                 self.defect_model, self.defect_model_type = self.load_model(model_path)
#                 self.defect_detection_enabled = True
#                 self.pushButton_defect.setText("关闭瑕疵检测")
#                 print("瑕疵检测已启用")
#             except Exception as e:
#                 print(f"加载瑕疵检测模型失败: {e}")
#         else:
#             self.defect_detection_enabled = False
#             self.defect_model = None
#             self.defect_model_type = None
#             self.pushButton_defect.setText("启用瑕疵检测")
#             print("瑕疵检测已关闭")
#
#         # 加载模型方法
#
#     def load_model(self, model_path):
#         """根据模型格式加载不同的模型"""
#         model_format = os.path.splitext(model_path)[1]
#         if model_format == '.pt':
#             # 使用SAHI的AutoDetectionModel加载模型
#             model = AutoDetectionModel.from_pretrained(
#                 model_type='yolov8',
#                 model_path=model_path,
#                 confidence_threshold=0.25,
#                 device="cuda:0" if torch.cuda.is_available() else "cpu"
#             )
#             return model, 'sahi'
#         else:
#             raise ValueError(f"不支持的模型格式: {model_format}")
#
#         # 瑕疵检测工作线程
#
#     def defect_detection_worker(self):
#         while True:
#             try:
#                 # 从队列获取图像数据
#                 image_data = self.defect_detection_queue.get(timeout=1)
#
#                 if not self.defect_detection_enabled or self.defect_model is None:
#                     continue
#
#                 # 处理图像
#                 image_path = image_data['image_path']
#                 pos_x = image_data['pos_x']
#                 pos_y = image_data['pos_y']
#
#                 # 使用SAHI进行切片检测
#                 result_img, sahi_result, object_count = self.process_image_with_sahi(
#                     self.defect_model,
#                     image_path,
#                     conf_threshold=0.25,
#                     slice_height=1024,
#                     slice_width=1024
#                 )
#
#                 if result_img is not None:
#                     # 保存结果图像
#                     result_filename = f"result_{pos_x}_{pos_y}.bmp"
#                     result_path = os.path.join(self.save_path_defect_results, result_filename)
#                     cv2.imwrite(result_path, result_img)
#
#                     # 如果有SAHI结果，导出可视化结果
#                     if sahi_result is not None:
#                         sahi_output_dir = os.path.join(self.save_path_defect_results, "sahi_visuals")
#                         if not os.path.exists(sahi_output_dir):
#                             os.makedirs(sahi_output_dir)
#                         sahi_result.export_visuals(export_dir=sahi_output_dir)
#
#                     # 将结果放入结果队列
#                     result_data = {
#                         'pos_x': pos_x,
#                         'pos_y': pos_y,
#                         'object_count': object_count,
#                         'result_path': result_path
#                     }
#                     self.defect_results_queue.put(result_data)
#
#                     print(f"位置 ({pos_x}, {pos_y}) 检测到 {object_count} 个瑕疵")
#
#             except queue.Empty:
#                 pass
#             except Exception as e:
#                 print(f"瑕疵检测线程错误: {e}")
#
#         # 处理图像方法
#
#     def process_image_with_sahi(self, model, image_path, conf_threshold=0.25, slice_height=1024, slice_width=1024):
#         """使用SAHI进行切片检测处理"""
#         try:
#             # 读取图片
#             frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#             if frame is None:
#                 print(f"无法读取图片: {image_path}")
#                 return None, None, 0
#
#             # 创建标注器
#             annotator = Annotator(frame, line_width=2)
#
#             # 使用SAHI进行切片预测
#             result = get_sliced_prediction(
#                 image_path,
#                 model,
#                 slice_height=slice_height,
#                 slice_width=slice_width,
#                 overlap_height_ratio=0.1,
#                 overlap_width_ratio=0.1,
#                 perform_standard_pred=True,
#                 postprocess_type="NMS",
#                 postprocess_match_threshold=0.5,
#                 postprocess_match_metric="IOU"
#             )
#
#             # 提取检测数据
#             detection_data = [
#                 (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
#                 for det in result.object_prediction_list
#             ]
#
#             # 统计瑕疵数量
#             cnt = 0
#             for det in detection_data:
#                 cnt += 1
#                 # 解析边界框坐标
#                 minx, miny, maxx, maxy = det[2]
#                 # 计算边界框的宽度和高度
#                 width = maxx - minx
#                 height = maxy - miny
#                 # 格式化宽度和高度，保留两位小数
#                 formatted_width = "{:.2f}".format(width)
#                 formatted_height = "{:.2f}".format(height)
#
#                 # 使用annotator在图像上绘制边界框和标签
#                 annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))
#
#             # 获取标注后的图像
#             result_img = annotator.result()
#             return result_img, result, cnt
#
#         except Exception as e:
#             print(f"处理图像时出错: {e}")
#             return None, None, 0
#
#     # 蛇形扫描
#     def scan_xy(self):
#         delaytime = 1
#         self.save_path_snakeScan += '//' + self.date_folder + '-' + self.time_folder
#         os.makedirs(self.save_path_snakeScan, exist_ok=True)
#         os.makedirs(self.save_path_snakeScan + '/焦前', exist_ok=True)
#         os.makedirs(self.save_path_snakeScan + '/焦后', exist_ok=True)
#         os.makedirs(self.save_path_snakeScan + '/焦面', exist_ok=True)
#
#         x_pos_start = self.doubleSpinBox_16.value()
#         y_pos_start = self.doubleSpinBox_17.value()
#         self.xy_pos = []
#         self.xy_pos = getposxy(x_pos_start, y_pos_start)
#
#         thread_xy_scan = threading.Thread(target=self.xy_scan_thread, args=(self.xy_pos,))
#         thread_xy_scan.start()
#
#     def xy_scan_thread(self, xy_pos):
#         try:
#             start_time = time.time()
#             self.status = 'xy_scan'
#
#             # print('xy_scan_thread')
#             # l=1
#
#             # self.g.GCommand(f'pPOSX={x_pos_start}')
#             self.g.GCommand(f'pPOSX={self.xy_pos[0][0]}')
#             self.g.GCommand('XQ  # MOVEX,1')
#
#             self.g.GCommand(f'pPOSY={self.xy_pos[0][1]}')
#             # self.g.GCommand(f'pPOSY={y_pos_start}')
#
#             self.g.GCommand('XQ  # MOVEY,2')
#             time.sleep(2)
#             self.timestatus = False
#             time.sleep(3)
#             self.get_image_flag_temp = 0
#             res_x = self.g.GCommand('TPA')
#             res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#             res_y = self.g.GCommand('TPB')
#             res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#             if res_x.strip():
#                 self.doubleSpinBox_100.setValue(int(res_x) / 6400)
#                 self.pos_x = int(res_x) / 6400
#             if res_y.strip():
#                 self.doubleSpinBox_101.setValue(int(res_y) / 6400)
#                 self.pos_y = int(res_y) / 6400
#
#             self.autofocus()
#             while self.autofocus_flag:  # 等待自动对焦完成
#                 pass
#             self.get_image_flag_temp = 0
#             self.pos_x = int(res_x) / 6400
#             self.pos_y = int(res_y) / 6400
#
#             self.status = 'snake_scan'
#             self.camera_save = 4
#             self.takePhoto()
#
#             while self.get_image_flag_temp < 3:
#                 pass
#             self.get_image_flag_temp = 0
#
#             for j in range(11):
#                 if self.stop_flag:
#                     break
#                 for i in range(11):
#                     if self.stop_flag:
#                         break
#
#                     print(xy_pos[i + j * 12 + 1][0])
#                     self.g.GCommand(f'pPOSX={self.xy_pos[i + j * 12 + 1][0]}')
#                     self.g.GCommand('XQ  # MOVEX,1')
#                     time.sleep(1)
#                     res_x = self.g.GCommand('TPA')
#                     res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                     res_y = self.g.GCommand('TPB')
#                     res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                     if res_x.strip():
#                         self.doubleSpinBox_100.setValue(int(res_x) / 6400)
#                         self.pos_x = int(res_x) / 6400
#                     if res_y.strip():
#                         self.doubleSpinBox_101.setValue(int(res_y) / 6400)
#                         self.pos_y = int(res_y) / 6400
#                     if self.pos_x and self.pos_y:
#                         self.pos_x = int(res_x) / 6400
#                         self.pos_y = int(res_y) / 6400
#                     self.autofocus()
#                     while self.autofocus_flag:  # 等待自动对焦完成
#                         pass
#
#                     self.status = 'snake_scan'
#                     self.camera_save = 4
#                     self.takePhoto()
#                     while self.get_image_flag_temp < 3:
#                         pass
#                     self.get_image_flag_temp = 0
#
#                     # print(xy_pos[i][0])
#                 print(xy_pos[j * 12 + 13][1])
#                 self.g.GCommand(f'pPOSY={self.xy_pos[j * 12 + 13][1]}')
#                 self.g.GCommand('XQ  # MOVEY,2')
#                 time.sleep(1)
#                 res_x = self.g.GCommand('TPA')
#                 res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                 res_y = self.g.GCommand('TPB')
#                 res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                 if res_x.strip():
#                     self.doubleSpinBox_100.setValue(int(res_x) / 6400)
#                     self.pos_x = int(res_x) / 6400
#                 if res_y.strip():
#                     self.doubleSpinBox_101.setValue(int(res_y) / 6400)
#                     self.pos_y = int(res_y) / 6400
#
#                 self.pos_x = int(res_x) / 6400
#                 self.pos_y = int(res_y) / 6400
#                 self.autofocus()
#                 while self.autofocus_flag:  # 等待自动对焦完成
#                     pass
#                 self.camera_save = 4
#                 self.status = 'snake_scan'
#                 self.takePhoto()
#                 while self.get_image_flag_temp < 3:
#                     pass
#                 self.get_image_flag_temp = 0
#             for k in range(11):
#
#                 if self.stop_flag:
#                     break
#                 # print(xy_pos[133+k][0])
#                 self.g.GCommand(f'pPOSX={self.xy_pos[133 + k][0]}')
#                 self.g.GCommand('XQ  # MOVEX,1')
#                 time.sleep(1)
#                 res_x = self.g.GCommand('TPA')
#                 res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                 res_y = self.g.GCommand('TPB')
#                 res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#                 if res_x.strip():
#                     self.doubleSpinBox_100.setValue(int(res_x) / 6400)
#                     self.pos_x = int(res_x) / 6400
#                 if res_y.strip():
#                     self.doubleSpinBox_101.setValue(int(res_y) / 6400)
#                     self.pos_y = int(res_y) / 6400
#
#                 self.pos_x = int(res_x) / 6400
#                 self.pos_y = int(res_y) / 6400
#                 self.autofocus()
#                 while self.autofocus_flag:  # 等待自动对焦完成
#                     pass
#
#                 self.status = 'snake_scan'
#                 self.camera_save = 4
#                 self.takePhoto()
#
#                 while self.get_image_flag_temp < 3:
#                     pass
#                 self.get_image_flag_temp = 0
#             self.timestatus = True
#             end_time = time.time()
#             print(f"蛇形扫描用时: {end_time - start_time} seconds")
#         except Exception as e:
#             print(e)
#
#     def autofocus(self):
#         self.autofocus_flag = True
#
#         self.takePhoto()
#         thread = threading.Thread(target=self.getAutoFocusImg)
#         thread.start()
#
#     def getAutoFocusImg(self):
#
#         while True:
#             try:
#                 item = self.autofocus_queue.get(timeout=2)
#                 if isinstance(item, np.ndarray):
#                     print(f'自动对焦图像：高{item.shape[0]},宽{item.shape[1]}')
#                     sleep(0.5)
#                     # 自动对焦逻辑
#
#                     break
#
#             except queue.Empty:
#                 pass
#         self.autofocus_flag = False
#         self.get_image_flag_temp = 0
#
#     # Z轴控制
#     def move_z_relative_up(self):
#         while True:
#             if self.is_press_up:
#                 self.is_stopping_z = False
#                 self.Zmc.ZAux_Direct_SetUserVar('relative_step', 1000)
#                 self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
#                 sleep(0.05)
#             else:
#                 break
#
#     def move_z_relative_down(self):
#
#         while True:
#             if self.is_press_down:
#                 self.is_stopping_z = False
#                 self.Zmc.ZAux_Direct_SetUserVar('relative_step', -1000)
#                 self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
#                 sleep(0.05)
#             else:
#                 break
#
#     def stop_z(self):
#         self.Zmc.ZAux_Direct_SetUserVar('relative_step', 0)
#         self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
#         self.is_stopping_z = True
#
#     # Z轴绝对运动
#     def move_z_up(self, pos):
#         self.is_stopping_z = False
#         pos = pos * 40000
#         self.Zmc.ZAux_Direct_SetUserVar('absolute_step', pos)
#         self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 2)
#
#     def onValueChanged_Zspeed(self, val):
#         speed = val * 600 / 1000
#         self.Zmc.ZAux_Direct_SetUserVar('z_speed', speed)
#         self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 5)
#
#     def move_z(self):
#         # print("2222")
#         a = self.doubleSpinBox_12.value()
#         self.move_z_up(a)
#
#     #    层扫
#     def zscan(self):
#
#         scan_up = self.doubleSpinBox_9.value()
#         scan_down = self.doubleSpinBox_14.value()
#         scan_step = self.doubleSpinBox_13.value()
#         base_path = "E:/ftkpic/layerscan"  # 指定基础路径
#         date_folder_path = os.path.join(base_path, self.date_folder)
#         os.makedirs(date_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略
#
#         # 创建时间文件夹
#         time_folder_path = os.path.join(date_folder_path, self.time_folder)
#         os.makedirs(time_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略
#
#         self.full_path = time_folder_path
#
#         # self.move_z_up(scan_up)
#         # self.is_stopping_z = False
#         thread_z_scan = threading.Thread(target=self.z_scan_thread, args=(scan_up, scan_down, scan_step,))
#         thread_z_scan.start()
#
#     def z_scan_thread(self, scan_up, scan_down, scan_step):
#         self.status = 'z_scan'
#         print('z_scan_thread')
#         self.move_z_up(scan_up)
#         while not self.is_stopping_z:
#             continue
#         self.camera_save = 1
#         while self.pos > scan_down:
#             self.move_z_up(self.pos / 40000 - scan_step)
#             while not self.is_stopping_z:
#                 continue
#             self.camera_save = 1
#         self.camera_save = 1
#
#     # 图片删除
#     def deletepic(self):
#         # 打开文件夹选择对话框
#         folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
#         if not folder_path:
#             return  # 如果用户取消选择，直接返回
#
#         # 确认是否删除
#         reply = QMessageBox.question(
#             self,
#             "确认删除",
#             f"确定要删除文件夹 '{folder_path}' 下的所有内容吗？",
#             QMessageBox.Yes | QMessageBox.No,
#             QMessageBox.No,
#         )
#         if reply == QMessageBox.No:
#             return  # 如果用户选择不删除，直接返回
#
#         # 删除文件夹内容
#         try:
#             self.delete_folder_contents(folder_path)
#             QMessageBox.information(self, "完成", "文件夹内容已删除！")
#         except Exception as e:
#             QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")
#
#     def delete_folder_contents(self, folder_path):
#         # 遍历文件夹内容并删除
#         for item in os.listdir(folder_path):
#             item_path = os.path.join(folder_path, item)
#             if os.path.isfile(item_path) or os.path.islink(item_path):
#                 os.unlink(item_path)  # 删除文件或符号链接
#             elif os.path.isdir(item_path):
#                 shutil.rmtree(item_path)  # 删除子文件夹及其内容
#
#     # 相机控制
#     def onValueChanged_cam1(self, val):
#         print(self.camList[0]['number'] + f'调整曝光时间：{val}')
#         ret = self.camList[0]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))
#
#     def onValueChanged_cam2(self, val):
#         print(self.camList[1]['number'] + f'调整曝光时间：{val}')
#         ret = self.camList[1]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))
#
#     def onValueChanged_cam3(self, val):
#         print(self.camList[2]['number'] + f'调整曝光时间：{val}')
#         ret = self.camList[2]['cam'].MV_CC_SetFloatValue("ExposureTime", float(val))
#
#     def takePhoto(self):
#
#         self.Zmc.ZAux_Direct_SetOp(0, 1)
#         self.Zmc.ZAux_Direct_SetOp(1, 1)
#         self.Zmc.ZAux_Direct_SetOp(2, 1)
#         sleep(0.01)
#         self.Zmc.ZAux_Direct_SetOp(0, 0)
#         self.Zmc.ZAux_Direct_SetOp(1, 0)
#         self.Zmc.ZAux_Direct_SetOp(2, 0)
#         # print('picture')
#
#     # 存图
#     def save_pic(self):
#         button = self.sender()
#         print(button.objectName())
#         self.status = 'trigger'
#         if button.objectName() == 'pushButton_20':  # 相机1触发存图
#             self.camera_save = 1
#         if button.objectName() == 'pushButton_18':  # 相机2触发存图
#             self.camera_save = 2
#         if button.objectName() == 'pushButton_19':  # 相机3触发存图
#             self.camera_save = 3
#         print(self.camera_save)
#
#     # XY载物台
#     def move_x(self):
#         print('')
#         a = self.doubleSpinBox_10.value()
#         self.g.GCommand(f'pPOSX={a}')
#         self.g.GCommand('XQ  # MOVEX,1')
#
#     def move_y(self):
#         print('')
#         a = self.doubleSpinBox_11.value()
#         self.g.GCommand(f'pPOSY={a}')
#         self.g.GCommand('XQ  # MOVEY,2')
#
#     def reset_x(self):
#         self.g.GCommand('XQ#HOMEX')
#
#     def reset_y(self):
#         self.g.GCommand('XQ#HOMEY')
#
#     # XY位置
#     def update_status(self):
#
#         # print(self.doubleSpinBox_10.value())
#         if self.timestatus:
#             res_x = self.g.GCommand('TPA')
#             res_x = res_x.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#             res_y = self.g.GCommand('TPB')
#             res_y = res_y.split(":")[-1].strip()  # 获取冒号后的部分并去除空白字符
#             if res_x.strip():
#                 self.doubleSpinBox_100.setValue(int(res_x) / 6400)
#                 self.pos_x = int(res_x) / 6400
#             if res_y.strip():
#                 self.doubleSpinBox_101.setValue(int(res_y) / 6400)
#                 self.pos_y = int(res_y) / 6400
#
#             if res_x and res_y:
#                 self.pos_x = int(res_x) / 6400
#                 self.pos_y = int(res_y) / 6400
#
#             self.takePhoto()
#
#         self.Zmc.ZAux_Direct_SetUserVar('g_cmd', 1)
#         pos = self.Zmc.ZAux_Direct_GetUserVar('local_position')[1].value
#         self.pos = pos
#         self.doubleSpinBox_102.setValue(int(pos) / 40000)
#         self.doubleSpinBox_104.setValue(int(pos) / 40000)
#
#         if len(self.z_pos_list) >= 3:
#             self.z_pos_list = self.z_pos_list[1:]
#         self.z_pos_list.append(pos)
#
#         if all(x == self.z_pos_list[0] for x in self.z_pos_list) and len(self.z_pos_list) >= 3:
#             self.is_stopping_z = True
#
#         now = datetime.now()
#         self.date_folder = now.strftime("%Y-%m-%d")  # 日期文件夹名称，例如 "2023-10-05"
#         self.time_folder = now.strftime("%H-%M-%S")  # 时间文件夹名称，例如 "14-30-00"
#
#         # if self.is_press:
#         #     Zmc.ZAux_Direct_SetUserVar('relative_step', 2000)
#         #     Zmc.ZAux_Direct_SetUserVar('g_cmd', 3)
#
#         # print(pos)
#
#     def camera_get_img(self, frame, number):
#         # 相机按钮取图
#         #         print(number)
#         self.get_image_flag_temp += 1
#         # if self.get_image_flag_temp>=3:
#         #     self.get_image_flag_temp = 0
#         if self.camera_save == 1:
#             if number == 'DA3827093':
#                 print(number)
#                 # cv2.imwrite(f'frame_{self.camera_save}.jpg', frame)
#                 now = datetime.now()
#                 # date_folder = now.strftime("%Y-%m-%d")
#                 # time_folder = now.strftime("%H-%M-%S")
#                 # Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦面\{date_folder+"-"+time_folder}.bmp")
#                 if self.status == 'trigger':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_triggerScan}\焦面\{self.date_folder + "-" + self.time_folder}.bmp")
#                 elif self.status == 'z_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.full_path}\{self.pos / 40000}.bmp")
#                 elif self.status == 'snake_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_snakeScan}\焦面\{self.pos_x}-{self.pos_y}.bmp")
#
#                 # cv2.imwrite(rf"E:\ftkpic\trigger\焦面\{timestamp}.jpg", frame)
#                 self.camera_save = 0
#         if self.camera_save == 2:
#             if number == 'DA3562117':
#                 print(number)
#                 now = datetime.now()
#                 date_folder = now.strftime("%Y-%m-%d")
#                 time_folder = now.strftime("%H-%M-%S")
#                 if self.status == 'trigger':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_triggerScan}\焦前\{self.date_folder + "-" + self.time_folder}.bmp")
#                 elif self.status == 'z_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.full_path}\{self.pos / 40000}.bmp")
#                 elif self.status == 'snake_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_snakeScan}\焦前\{self.pos_x}-{self.pos_y}.bmp")
#
#                 self.camera_save = 0
#         if self.camera_save == 3:
#             if number == 'DA3562103':
#                 print(number)
#                 now = datetime.now()
#                 date_folder = now.strftime("%Y-%m-%d")
#                 time_folder = now.strftime("%H-%M-%S")
#                 if self.status == 'trigger':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_triggerScan}\焦后\{self.date_folder + "-" + self.time_folder}.bmp")
#                 elif self.status == 'z_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.full_path}\{self.pos / 40000}.bmp")
#                 elif self.status == 'snake_scan':
#                     Image.fromarray(frame).save(
#                         rf"{self.save_path_snakeScan}\焦后\{self.pos_x}-{self.pos_y}.bmp")
#
#                 self.camera_save = 0
#                 self.status = None
#         if self.camera_save == 4:
#             self.camera_save_index += 1
#             if self.status == 'snake_scan':
#
#                 pic_path = ""
#
#                 if number == 'DA3562103':
#                     pic_path = rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp"
#                     Image.fromarray(frame).save(pic_path)
#
#                 if number == 'DA3562117':
#                     pic_path = rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp"
#                     Image.fromarray(frame).save(pic_path)
#
#                 if number == 'DA3827093':
#                     pic_path = rf"{self.save_path_snakeScan}\焦面\{self.pos_x},{self.pos_y}.bmp"
#                     Image.fromarray(frame).save(pic_path)
#
#                     # 对焦平面图像进行瑕疵检测
#                     if self.defect_detection_enabled and self.defect_model is not None:
#                         # 将图像添加到瑕疵检测队列
#                         self.defect_detection_queue.put({
#                             'image_path': pic_path,
#                             'pos_x': self.pos_x,
#                             'pos_y': self.pos_y
#                         })
#
#                 if self.camera_save_index >= 3:
#                     self.camera_save_index = 0
#                     self.camera_save = 0
#                     self.status = None
#
#         if not self.timestatus:
#             # print('计时器状态更新停止')
#             if self.autofocus_flag:
#                 if number == 'DA3827093':
#                     self.autofocus_queue.put(frame)  # 传递焦面图片到自动对焦函数
#             else:
#                 # self.image_data[number]=frame
#                 if number == 'DA3562103':
#                     pic_path = rf"{self.save_path_snakeScan}\焦后\{self.pos_x},{self.pos_y}.bmp"
#                 if number == 'DA3562117':
#                     pic_path = rf"{self.save_path_snakeScan}\焦前\{self.pos_x},{self.pos_y}.bmp"
#                 if number == 'DA3827093':
#                     pic_path = rf"{self.save_path_snakeScan}\焦面\{self.pos_x},{self.pos_y}.bmp"
#
#                 self.image_data[number] = {
#                     'frame': frame,
#                     'pic_path': pic_path
#                 }
#                 if len(self.image_data) >= 3:
#                     # print('图片传到图像处理进程')
#                     #
#                     self.debug_image_queue.put(self.image_data)
#                     self.image_data = {}
#         # 三个相机显示
#         height, width = frame.shape
#         q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
#         if number == 'DA3827093':
#             self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#         if number == 'DA3562117':
#             self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#         if number == 'DA3562103':
#             self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#             # cv2.imwrite('frame.jpg',frame)
#     # def camera_get_img(self, cam, pData=0, nDataSize=0,number=0):  # 相机取图线程
#     #     stFrameInfo = MV_FRAME_OUT_INFO_EX()
#     #     memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
#     #     while True:
#     #         # print('wait')
#     #         try:
#     #             ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 10000)
#     #             if ret == 0:
#     #                 # print("get one frame: Width[%d], Height[%d], nFrameNum[%d], number[%s]" % (
#     #                 #     stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum,number))
#     #
#     #                 frame = np.array(pData)  # 将c_ubyte_Array转化成ndarray得到（5308416，）
#     #                 frame = cv2.resize(frame, (2048, 2048))
#     #                 height, width = frame.shape
#     #                 q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
#     #                 # self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 # self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 # self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 # 相机按钮取图
#     #                 if self.camera_save == 1:
#     #                     if number == 'DA3827093':
#     #                         print(number)
#     #                         # cv2.imwrite(f'frame_{self.camera_save}.jpg', frame)
#     #                         now = datetime.now()
#     #                         date_folder = now.strftime("%Y-%m-%d")
#     #                         time_folder = now.strftime("%H-%M-%S")
#     #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦面\{date_folder+"-"+time_folder}.bmp")
#     #                         # cv2.imwrite(rf"E:\ftkpic\trigger\焦面\{timestamp}.jpg", frame)
#     #                         self.camera_save = 0
#     #                 if self.camera_save == 2:
#     #                     if number == 'DA3562117':
#     #                         print(number)
#     #                         now = datetime.now()
#     #                         date_folder = now.strftime("%Y-%m-%d")
#     #                         time_folder = now.strftime("%H-%M-%S")
#     #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦前\{date_folder+"-"+time_folder}.bmp")
#     #                         self.camera_save = 0
#     #                 if self.camera_save == 3:
#     #                     if number == 'DA3562103':
#     #                         print(number)
#     #                         now = datetime.now()
#     #                         date_folder = now.strftime("%Y-%m-%d")
#     #                         time_folder = now.strftime("%H-%M-%S")
#     #                         Image.fromarray(frame).save(rf"E:\ftkpic\trigger\焦后\{date_folder+"-"+time_folder}.bmp")
#     #                         self.camera_save = 0
#     #                 # 三个相机显示
#     #                 if number == 'DA3827093':
#     #                     self.label_9.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 if number == 'DA3562117':
#     #                    self.label_10.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 if number == 'DA3562103':
#     #                    self.label_8.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #                 # cv2.imwrite('frame.jpg',frame)
#     #
#     #                 # cv2.imwrite("D:/123456/frame%d.jpg" % stFrameInfo.nFrameNum, frame)
#     #                 # height, width = frame.shape
#     #
#     #                 # 将OpenCV灰度图像转换为QImage
#     #                 # 单通道图像需要使用不同的QImage.Format
#     #                 # if self.ui:
#     #                 #     q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
#     #                 #     self.ui.imageLabel.setPixmap(QPixmap.fromImage(q_img).scaled(width // 4, height // 4))
#     #
#     #                 # cv2.imshow("temp", temp)
#     #                 # cv2.waitKey(0)
#     #
#     #         except Exception as e:
#     #             print('相机报错:', e)
#
#
# # 主程序初始化
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     window = Debug_UI(mode='test')
#     sys.exit(app.exec_())