# import threading
#
# import cv2
# import numpy as np
# from PyQt5.QtGui import QImage, QPixmap
#
# from window.control.MvImport.CameraParams_header import MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO, MV_TRIGGER_MODE_ON, \
#     MVCC_INTVALUE, MV_FRAME_OUT_INFO_EX
# from window.control.MvImport.MvCameraControl_class import *
#
# class CameraFactory:
#     def __init__(self,ui=None):
#         self.ui = ui
#         self.cams = []
#         deviceList = MV_CC_DEVICE_INFO_LIST()
#         tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
#
#         # ch:枚举设备 | en:Enum device
#         ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
#         if ret != 0:
#             print("enum devices fail! ret[0x%x]" % ret)
#             # sys.exit()
#             # openCamera(det_queue,win_queue)
#
#         if deviceList.nDeviceNum == 0:
#             print("find no device!")
#             # sys.exit()
#
#         print("find %d devices!" % deviceList.nDeviceNum)
#
#         for i in range(0, deviceList.nDeviceNum):
#             mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
#             if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
#                 print("\ngige device: [%d]" % i)
#                 strModeName = ""
#                 for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
#                     strModeName = strModeName + chr(per)
#                 print("device model name: %s" % strModeName)
#
#                 nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
#                 nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
#                 nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
#                 nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
#                 print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
#             elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
#                 print("\nu3v device: [%d]" % i)
#                 strModeName = ""
#                 for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
#                     if per == 0:
#                         break
#                     strModeName = strModeName + chr(per)
#                 print("device model name: %s" % strModeName)
#
#                 strSerialNumber = ""
#                 for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
#                     if per == 0:
#                         break
#                     strSerialNumber = strSerialNumber + chr(per)
#                 print("user serial number: %s" % strSerialNumber)
#
#                 # ch:创建相机实例 | en:Creat Camera Object
#                 cam = MvCamera()
#
#                 # ch:选择设备并创建句柄 | en:Select device and create handle
#                 stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
#
#                 ret = cam.MV_CC_CreateHandle(stDeviceList)
#                 if ret != 0:
#                     print("create handle fail! ret[0x%x]" % ret)
#                     # sys.exit()
#
#                 # ch:打开设备 | en:Open device
#                 ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
#                 if ret != 0:
#                     print("open device fail! ret[0x%x]" % ret)
#                     # sys.exit()
#
#                 # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
#                 if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
#                     nPacketSize = cam.MV_CC_GetOptimalPacketSize()
#                     if int(nPacketSize) > 0:
#                         ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
#                         if ret != 0:
#                             print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
#
#                     else:
#                         print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
#
#
#                 # ch:设置触发模式为on| en:Set trigger mode as on
#                 ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
#                 if ret != 0:
#                     print("set trigger mode fail! ret[0x%x]" % ret)
#
#                 ret = cam.MV_CC_SetFloatValue("ExposureTime", 1000)
#
#
#                 ret = cam.MV_CC_SetEnumValue("TriggerSource", 0)
#                 if ret != 0:
#                     print("set trigger mode fail! ret[0x%x]" % ret)
#
#
#                 ret = cam.MV_CC_SetEnumValue("LineSelector", 1)
#                 if ret != 0:
#                     print("set LineSelecto_NETr fail! ret[0x%x]" % ret)
#
#
#                 # DurationValue, DelayValue , PreDelayValue = 0,0,0
#                 # ret = cam.MV_CC_SetIntValue("StrobeLineDuration", 0)
#                 # if ret != 0:
#                 #     print("set StrobeLineDuration fail! ret[0x%x]" % ret)
#                 #
#                 # ret = cam.MV_CC_SetIntValue("StrobeLineDelay", 0)
#                 # if ret != 0:
#                 #     print("set StrobeLineDelay fail! ret[0x%x]" % ret)
#                 #
#                 # ret = cam.MV_CC_SetIntValue("StrobeLinePreDelay", 0)
#                 # if ret != 0:
#                 #     print("set StrobeLinePreDelay fail! ret[0x%x]" % ret)
#                 #
#                 # ret = cam.MV_CC_SetBoolValue("StrobeEnable", True)
#                 # if ret != 0:
#                 #     print("set StrobeEnable fail! ret[0x%x]" % ret)
#
#                 # nRet = cam.MV_CC_SetImageNodeNum(100)
#                 # ch:获取数据包大小 | en:Get payload size
#                 stParam = MVCC_INTVALUE()
#                 memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
#
#                 ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
#                 if ret != 0:
#                     print("get payload size fail! ret[0x%x]" % ret)
#
#                 nPayloadSize = stParam.nCurValue
#
#                 # ch:开始取流 | en:Start grab image
#                 # ret = cam.MV_CC_StartGrabbing(self.ui.label_6.winId())
#                 # obj_cam_operation = CameraOperation(cam, deviceList, 0)
#                 ret = cam.MV_CC_StartGrabbing()
#                 if ret != 0:
#                     print("start grabbing fail! ret[0x%x]" % ret)
#
#                 data_buf = (c_ubyte * nPayloadSize)()
#                 # label = camera_label_list[nConnectionNum]
#                 elements = [cam, data_buf, nPayloadSize,strSerialNumber]
#                 self.cams.append(elements)
#                 # try:
#                 #
#                 #     hThreadHandle = threading.Thread(target=self.work_thread,
#                 #                                      args=(
#                 #                                          cam, data_buf, nPayloadSize,
#                 #                                          ))
#                 #     hThreadHandle.start()
#                 #
#                 #
#                 # except:
#                 #     print("error: unable to start thread")
#
#
#     # def work_thread(self, cam, pData=0, nDataSize=0):  # 相机取图线程
#     #     stFrameInfo = MV_FRAME_OUT_INFO_EX()
#     #     memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
#     #     while True:
#     #
#     #         try:
#     #             ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 10000)
#     #             if ret == 0:
#     #                 print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
#     #                     stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
#     #
#     #                 frame = np.array(pData)  # 将c_ubyte_Array转化成ndarray得到（5308416，）
#     #                 frame = cv2.resize(frame, (2048, 2048))
#     #
#     #                 #
#     #                 # cv2.imwrite("D:/123456/frame%d.jpg" % stFrameInfo.nFrameNum, frame)
#     #                 # height, width = frame.shape
#     #                 #
#     #                 # # 将OpenCV灰度图像转换为QImage
#     #                 # # 单通道图像需要使用不同的QImage.Format
#     #                 # q_img = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
#     #                 #
#     #                 # self.ui.imageLabel.setPixmap(QPixmap.fromImage(q_img).scaled(width//4, height//4))
#     #
#     #                 # cv2.imshow("temp", temp)
#     #                 # cv2.waitKey(0)
#     #
#     #         except Exception as e:
#     #             print('相机报错:', e)
#
#     def setCameraCallback(self,func):
#         for element in self.cams:
#             try:
#                 hThreadHandle = threading.Thread(target=func,
#                                                  args=(
#                                                      element[0], element[1], element[2],element[3]
#                                                  ))
#                 hThreadHandle.start()
#             except:
#                 print("error: unable to start thread")
#
#
#
#
# if __name__ == '__main__':
#     camera = CameraFactory()
import time

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal,QObject
try:
    from pypylon import pylon
except ImportError:
    print("警告：pypylon库未安装，相机功能将不可用")
from .duiao import LightSpotAnalyzer

from window.control.MvImport.CameraParams_header import *
from window.control.MvImport.MvCameraControl_class import *
from PyQt5.QtCore import QMutex
class CameraWorker(QThread):
    image_signal = pyqtSignal(np.ndarray,str)
    set_trigger_mode_signal = pyqtSignal(int)

    def __init__(self, cam_info,camera_id, parent=None):
        super(CameraWorker, self).__init__(parent)
        self.cam = MvCamera()
        self.st_device_info = cam_info
        self.camera_id = camera_id
        self.running = True
        self.set_trigger_mode_signal.connect(self.set_trigger_mode)


    def set_trigger_mode(self, mode):
        """设置触发模式（0=关闭，1=开启）"""
        if not mode:
            # print('连续模式')
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                print(f"关闭失败，错误码：{ret}")
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", mode)
            if ret != 0:
                print(f"设置触发模式失败，错误码：{ret}")
            ret = self.cam.MV_CC_StartGrabbing()
        else:
            # print('硬触发模式')
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                print(f"关闭失败，错误码：{ret}")
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", 1)
            ret = self.cam.MV_CC_SetEnumValue("TriggerSource", 0)  # 硬件触发源
            if ret != 0:
                print(f"设置触发模式失败，错误码：{ret}")
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                print(f"打开失败，错误码：{ret}")

        stEnum = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("TriggerMode", stEnum)
        if ret == 0:
            print("当前触发模式值：", stEnum.nCurValue)
        else:
            print("获取失败，错误码：", ret)
            # print(self.running)


    def run(self):

        # self.running = True
        ret = self.cam.MV_CC_CreateHandle(self.st_device_info)
        if ret != 0:
            print("创建句柄失败")
            return

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("打开设备失败")
            return

        # print(self.changeModeFlag)
        if self.camera_id != '00E75147863':
            # 设置为硬触发模式
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", 1)  # 开启触发模式
            ret = self.cam.MV_CC_SetEnumValue("TriggerSource", 0)  # 硬件触发源
            # print(ret)
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 3000)  # 初始曝光值为1000
        else:
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 1000)  # 初始曝光值为1000

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("开始取流失败")
            return



        while self.running:
            st_out_frame = MV_FRAME_OUT()
            memset(byref(st_out_frame), 0, sizeof(st_out_frame))
            ret = self.cam.MV_CC_GetImageBuffer(st_out_frame, 1000)
            if ret == 0 and st_out_frame.pBufAddr is not None:
                img_buff = (c_ubyte * st_out_frame.stFrameInfo.nFrameLen)()
                cdll.msvcrt.memcpy(byref(img_buff), st_out_frame.pBufAddr, st_out_frame.stFrameInfo.nFrameLen)
                data_type = np.uint8
                img_array = np.frombuffer(img_buff, dtype=data_type)
                img = img_array.reshape((st_out_frame.stFrameInfo.nHeight, st_out_frame.stFrameInfo.nWidth))
                # print('frame')
                self.image_signal.emit(img,self.camera_id)
                self.cam.MV_CC_FreeImageBuffer(st_out_frame)

            # print('取图结束')
            # self.cam.MV_CC_StopGrabbing()
            # self.cam.MV_CC_CloseDevice()
            # self.cam.MV_CC_DestroyHandle()




    def stop(self):
        self.running = False

class CameraFactory:
    def __init__(self):
        self.cams = []
        self.workers = []
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            print("没有找到设备")
            return
        if device_list.nDeviceNum == 0:
            print("find no device!")
            return

        print("Find %d camera devices!" % device_list.nDeviceNum)


        for i in range(device_list.nDeviceNum):

            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                # print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                # print("device model name: %s" % strModeName)
                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)

                worker = CameraWorker(mvcc_dev_info,strSerialNumber)
                self.cams.append({
                    'cam':worker.cam,
                    'number':strSerialNumber
                })
                # worker.image_signal.connect(func)
                self.workers.append(worker)
                # worker.start


    def setFunc(self,func):
        for worker in self.workers:
            worker.image_signal.connect(func)
            worker.start()
        # time.sleep(2)
        # self.setMode()

        return self.cams

    def setMode(self,mode):
        for worker in self.workers:
            worker.set_trigger_mode_signal.emit(mode)



class BSLCameraWorker(QObject):
    """相机采集工作线程类"""
    frame_ready = pyqtSignal(np.ndarray, int)  # 图像帧信号 (图像数据, 相机ID)
    error_occurred = pyqtSignal  # 错误信号 (错误信息, 相机ID)

    def __init__(self, serial_number, camera_id):
        super().__init__()
        self.serial_number = serial_number  # 相机序列号
        self.camera_id = camera_id  # 相机ID (1或2)
        self.exposure_time = 10000  # 默认曝光时间 (µs)
        self.running = True  # 运行状态标志
        self.camera = None  # 相机实例
        self.converter = None  # 图像格式转换器


    def init_camera(self):
        """初始化相机连接及参数"""
        print('aaa')
        # try:
        # 检查pylon库是否可用
        if 'pylon' not in globals():
            self.error_occurred.emit("pypylon 库未安装，无法使用相机功能", self.camera_id)
            return False

            # 通过序列号查找相机
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        target_device = None
        for dev in devices:
            if dev.GetSerialNumber() == self.serial_number:
                target_device = dev
                break

        if not target_device:
            self.error_occurred.emit(f" 未找到序列号为 {self.serial_number}  的相机", self.camera_id)
            return False

            # 创建相机实例并配置参数
        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(target_device))
        self.camera.Open()
        self.camera.ExposureAuto.SetValue('Off')  # 关闭自动曝光
        self.camera.ExposureMode.SetValue('Timed')  # 设置定时曝光模式
        self.set_exposure(self.exposure_time)  # 应用曝光时间

        # 初始化图像格式转换器 (转为BGR8格式)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # 开始图像采集 (仅保留最新帧)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(
            f"相机{self.camera_id} 已连接: {self.camera.GetDeviceInfo().GetModelName()}  (SN: {self.serial_number})")

        return True

        # except Exception as e:
        #     self.error_occurred.emit(f" 相机{self.camera_id} 初始化错误: {str(e)}", self.camera_id)
        #     return False

    def set_exposure(self, exposure_time):
        """设置相机曝光时间 (µs)"""
        self.exposure_time = exposure_time
        try:
            if self.camera:
                # 根据相机型号适配曝光时间参数名
                if hasattr(self.camera, 'ExposureTime'):
                    self.camera.ExposureTime.SetValue(exposure_time)
                elif hasattr(self.camera, 'ExposureTimeAbs'):
                    self.camera.ExposureTimeAbs.SetValue(exposure_time)
        except Exception as e:
            self.error_occurred.emit(f" 设置曝光错误: {str(e)}", self.camera_id)

    def run(self):
        """相机采集主循环"""

        if not self.init_camera():
            return

        while self.running:
            try:
                if self.camera is None or not self.camera.IsGrabbing():
                    self.error_occurred.emit(" 相机未初始化或已停止采集", self.camera_id)
                    QtCore.QThread.msleep(100)
                    continue

                    # 获取图像帧 (超时100ms)
                grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
                if grab_result.GrabSucceeded():
                    # 转换图像格式并发送信号
                    image = self.converter.Convert(grab_result)
                    frame = image.GetArray()
                    self.frame_ready.emit(frame, self.serial_number)
                grab_result.Release()  # 释放帧资源

            except Exception as e:
                self.error_occurred.emit(f" 图像采集错误: {str(e)}", self.camera_id)
                QtCore.QThread.msleep(100)

    def stop(self):
        """停止相机采集并释放资源"""
        self.running = False
        if self.camera:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self.camera.Close()
