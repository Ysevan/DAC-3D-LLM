import ctypes
import platform

# 运行环境判断
systype = platform.system()
if systype == 'Windows':
    if platform.architecture()[0] == '64bit':
        zauxdll = ctypes.WinDLL('./zauxdll.dll')
        # print('Windows x64')
    else:
        zauxdll = ctypes.WinDLL('./zauxdll.dll')
        print('Windows x86')
elif systype == 'Darwin':
    zmcdll = ctypes.CDLL('./zmotion.dylib')
    print("macOS")
elif systype == 'Linux':
    zmcdll = ctypes.CDLL('./zmotion.so')
    print("Linux")
else:
    print("OS Not Supported!!")


class ZAUXDLL:
    def __init__(self):
        self.handle = ctypes.c_void_p()

    def ZAux_Execute(self, pszCommand):
        '''
        :Description:封装 Excute 函数, 以便接收错误。       
        :param pszCommand:字符串命令。              type: sting

        :param uiResponseLength:返回的字符长度。    type: uint32

        :Return:错误码,返回的字符串。              type: int32,sting

       '''
        _str = pszCommand.encode('utf-8')
        psResponse = ctypes.c_char_p()
        psResponse.value = b''
        uiResponseLength = 2048
        ret = zauxdll.ZAux_Execute(self.handle, _str, psResponse, uiResponseLength)
        rev = psResponse.value.decode('utf-8')
        return ret, rev
    

    def ZAux_DirectCommand(self, pszCommand):
        '''
        :Description:封装 DirectCommand 函数, 以便接收错误。

        :param pszCommand:字符串命令。              type: sting

        :param uiResponseLength:返回的字符长度。    type: uint32

        :Return:错误码,返回的字符串。              type: int32,sting

       '''
        _str = pszCommand.encode('utf-8')
        psResponse = ctypes.c_char_p()
        psResponse.value = b''
        uiResponseLength = 2048
        ret = zauxdll.ZAux_DirectCommand(self.handle, _str, psResponse, uiResponseLength)
        rev = psResponse.value.decode('utf-8')
        return ret, rev

    def ZAux_OpenEth(self, ipaddr):
        '''
        :Description:与控制器建立链接。

        :param ipaddress:IP地址,字符串的方式输入。   type: sting

        :Return:错误码。                            type: int32

       '''
        ip_bytes = ipaddr.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        return ret

    def ZAux_SearchEthlist(self, address_buff_length, ms):
        #  '''
        #  :Description:搜索当前网段下的 IP 地址。。
        #
        #  :param addrbufflength:搜索返回的 IP 地址总长度  type: uint32
        #
        #  :param ms:搜索超时时间。              type: uint32
        #
        #  :Return,Ipaddrlist:错误码, 搜索到的全部 IP 地址   type: sting
        #
        # '''

        # ip_address_list = ctypes.c_char_p(ip_address_list.encode('utf-8'))
        address_buff_length = ctypes.c_uint32(address_buff_length)
        ms = ctypes.c_uint32(ms)
        ip = ctypes.c_char_p("".encode('utf-8'))
        ret = zauxdll.ZAux_SearchEthlist(ip, address_buff_length, ms)
        return ret, ip

    def ZAux_SearchEth(self, ipaddress, uims):
        #  '''
        #  :Description:快速检索控制器。
        #
        #  :param ipaddress:控制器IP地址。  type: sting
        #
        #  :param uims:响应时间。              type: uint32
        #
        #  :Return:错误码, ERR_OK表示有搜索到。 type: int32
        #
        # '''
        ip_bytes = ipaddress.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_SearchEth(p_ip, ctypes.c_int(uims), ctypes.pointer(self.handle))
        return ret

    def ZAux_OpenCom(self, comid):
        '''
        :Description:与控制器建立链接，串口方式。

        :param comid:串口号   type: uint32

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_OpenCom(ctypes.c_uint32(comid), ctypes.pointer(self.handle))
        return ret

    def ZAux_Close(self):
        '''
        :Description:关闭控制器链接。

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Close(self.handle)
        return ret

    def ZAux_Direct_GetAD(self, ionum):
        '''
        :Description:读取模拟量输入信号。

        :param ionum:AIN口编号。   type: int

        :Return:错误码,返回的模拟量值 4系列以下0-4095。 type: int32,folat

       '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetAD(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDA(self, ionum, fValue):
        '''
        :Description:打开模拟量输出信号。

        :param ionum:DA输出口编号。  type: int

        :param fValue:设定的模拟量值4系列以下0-4095。  type: float

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Direct_SetDA(self.handle, ctypes.c_int(ionum), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetDA(self, ionum):
        '''
        :Description:读取模拟输出口状态。

        :param ionum:模拟量输出口编号。   type: int

        :Return:错误码,返回的模拟量值 4系列以下0-4095。 type: int32, float

       '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDA(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_SearchAndOpenCom(self, uimincomidfind, uimaxcomidfind, uims):
        '''
        :Description:快速控制器建立链接。

        :param uimincomidfind:最小串口号。   type: uint32

        :param uimincomidfind:最大串口号。   type: uint32

        :param uims:链接时间。   type: uint32

        :Return:错误码,有效COM,卡链接handle 。 type: int32,uint

       '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_SearchAndOpenCom(ctypes.c_uint32(uimincomidfind), ctypes.c_uint32(uimaxcomidfind),
                                            ctypes.byref(value), uims, ctypes.pointer(self.handle))
        return ret, value

    def ZAux_OpenEth(self, ipaddr):
        '''
        :Description:与控制器建立链接。

        :param ipaddr:IP地址,字符串的方式输入。   type: sting

        :Return:错误码。 type: int32

       '''
        ip_bytes = ipaddr.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        return ret

    def ZAux_SetComDefaultBaud(self, dwBaudRate, dwByteSize, dwParity, dwStopBits):
        '''
        :Description:可以修改缺省的波特率等设置。

        :param dwBaudRate:波特率。   type: uint32

        :param dwParity:NOPARITY,校验位。   type: uint32

        :param dwStopBits:ONESTOPBIT停止位。   type: uint32

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_SetComDefaultBaud(ctypes.c_uint32(dwBaudRate), ctypes.c_uint32(dwByteSize),
                                             ctypes.c_uint32(dwParity), ctypes.c_uint32(dwStopBits))
        return ret

    def ZAux_SetIp(self, ipaddress):
        '''
        :Description:修改控制器IP地址。

        :param ipaddress:IP地址。   type: sting

        :Return:错误码。 type: int32

       '''
        ip_bytes = ipaddress.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_SetIp(self.handle, p_ip)
        return ret

    def ZAux_Resume(self):
        '''
        :Description:暂停继续运行BAS项目。

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Resume(self.handle)
        return ret

    def ZAux_Pause(self):
        '''
        :Description:暂停控制器中BAS程序

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Pause(self.handle)
        return ret

    def ZAux_BasDown(self, Filename, run_mode):
        '''
        :Description:单个BAS文件生成ZAR并且下载到控制器运行。

        :param Filename:BAS文件路径。   type: sting

        :param run_mode:0-RAM  1-ROM。   type: uint32

        :Return:错误码。 type: int32

       '''
        _str = Filename.encode('utf-8')
        ret = zauxdll.ZAux_BasDown(self.handle, _str, run_mode, ctypes.pointer(self.handle))
        return ret

    def ZAux_Direct_GetIn(self, ionum):
        '''
        :Description:读取输入信号。

        :param ionum:IN编号。   type: int

        :Return:错误码,输入口状态。 type: int32, uint32

       '''
        value = ctypes.c_int32()
        ret = zauxdll.ZAux_Direct_GetIn(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetOp(self, ionum, iValue):
        '''
        :Description:打开输出信号。

        :param ionum:输出口编号。   type: int

        :param iValue:输出口状态。   type: uint32

        :Return:错误码。 type: int32

       '''

        ret = zauxdll.ZAux_Direct_SetOp(self.handle, ionum, iValue)
        return ret

    def ZAux_Direct_GetOp(self, ionum):
        '''
        :Description:读取输出口状态。

        :param ionum:输出口编号。   type: int

        :Return:错误码,输出口状态。 type: int32,uint32

       '''
        value = ctypes.c_int32()
        ret = zauxdll.ZAux_Direct_GetOp(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetInvertIn(self, ionum, bifInvert):
        '''
        :Description:设置输入口反转。

        :param ionum:输出口编号。   type: int

        :param bifInvert:反转状态 0/1。  type: int


        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Direct_SetInvertIn(self.handle, ionum, bifInvert)
        return ret

    def ZAux_Direct_GetInvertIn(self, ionum):
        '''
        :Description:读取输入口反转状态。

        :param ionum:输出口编号。   type: int

        :Return:错误码,反转状态。 type: int32 ,int

       '''
        value = ctypes.c_int32()
        ret = zauxdll.ZAux_Direct_GetInvertIn(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetPwmFreq(self, ionum, fValue):
        '''
        :Description:设置pwm频率。

        :param ionum:PWM编号口。   type: int

        :param fValue:频率 硬件PWM1M 软PWM 2K。    type: float

        :Return:错误码。 type: int32

       '''
        ret = zauxdll.ZAux_Direct_SetPwmFreq(self.handle, ctypes.c_int(ionum), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetPwmDuty(self, ionum, fValue):
        '''
        :Description:设置pwm占空比。

        :param ionum:PWM编号口。   type: int

        :param fValue:占空变	0-1  0表示关闭PWM口。    type: float

        :Return:错误码。 type: int32

       '''

        ret = zauxdll.ZAux_Direct_SetPwmDuty(self.handle, ctypes.c_int(ionum), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetPwmDuty(self, ionum):
        '''
        :Description:设置pwm占空比。

        :param ionum:PWM编号口。   type: int

        :Return:错误码,返回的空比。 type: int32,float

       '''

        ret = zauxdll.ZAux_Direct_SetPwmDuty(self.handle, ctypes.c_int(ionum))
        return ret

    def ZAux_Direct_GetPwmFreq(self, ionum):
        '''
        :Description:读取pwm频率。

        :param ionum:PWM编号口。   type: int

        :Return:错误码,返回的频率。 type: int32,float

       '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetPwmFreq(self.handle, ctypes.c_int(ionum), ctypes.byref(value))
        return ret, value

    def ZAux_GetModbusIn(self, ionumfirst, ionumend):
        '''
        :Description:参数 快速读取多个输入。

        :param ionumfirst:IN起始编号。   type: int

        :param ionumend:IN结束编号。   type: int

        :Return:错误码,位状态按位存储。 type: int32,uint8

       '''
        value = ctypes.c_int8()
        ret = zauxdll.ZAux_GetModbusIn(self.handle, ctypes.c_int(ionumfirst), ctypes.c_int(ionumend),
                                       ctypes.byref(value))
        return ret, value

    def ZAux_GetModbusOut(self, ionumfirst, ionumend):
        '''
        :Description:参数 快速读取多个当前的输出状态。

        :param ionumfirst:IN起始编号。   type: int

        :param ionumend:IN结束编号。   type: int

        :Return:错误码,位状态按位存储。 type: int32,uint8

        '''
        value = ctypes.c_int8()
        ret = zauxdll.ZAux_GetModbusOut(self.handle, ctypes.c_int(ionumfirst), ctypes.c_int(ionumend),
                                        ctypes.byref(value))
        return ret, value

    def ZAux_GetModbusDpos(self, imaxaxises):
        '''
        :Description:参数 快速读取多个当前的DPOS。

        :param imaxaxises:轴数量   type: int

        :Return:错误码,读取的坐标值从轴0开始。 type: int32,float

        '''
        value = (ctypes.c_float * imaxaxises)()
        ret = zauxdll.ZAux_GetModbusDpos(self.handle, imaxaxises, value)
        return ret, value

    def ZAux_GetModbusMpos(self, imaxaxises):
        '''
        :Description:参数 快速读取多个当前的MPOS。

        :param imaxaxises:轴数量   type: int

        :Return:错误码, 读取的反馈坐标值 从轴0开始。 type: int32,float

        '''
        value = (ctypes.c_float * imaxaxises)()
        ret = zauxdll.ZAux_GetModbusMpos(self.handle, imaxaxises, value)
        return ret, value

    def ZAux_GetModbusCurSpeed(self, imaxaxises):
        '''
        :Description:参数 快速读取多个当前的速度。

        :param imaxaxises:轴数量   type: int

        :Return:错误码,读取的当前速度 从轴0开始。 type: int32,float

        '''
        value = (ctypes.c_float * imaxaxises)()
        ret = zauxdll.ZAux_GetModbusCurSpeed(self.handle, imaxaxises, value)
        return ret, value

    def ZAux_Direct_SetAccel(self, iaxis, fValue):
        '''
        :Description:设置加速度。

        :param iaxis:轴号   type: int

        :param fValue:设定值   type: float

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetAccel(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetParam(self, sParam, iaxis, fset):
        '''
        :Description:通用的参数修改函数 sParam: 填写参数名称。

        :param sParam:轴参数名称 "DPOS" ...   type: sting

        :param iaxis:轴号   type: int

        :param fset:设定值   type: float

        :Return:错误码。 type: int32

        '''
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_SetParam(self.handle, _str, ctypes.c_int(iaxis), ctypes.c_float(fset))
        return ret

    def ZAux_Direct_GetParam(self, sParam, iaxis):
        '''
        :Description:通参数 通用的参数读取函数, sParam:填写参数名称。

        :param sParam:轴参数名称 "DPOS" ...   type: sting

        :param iaxis:轴号   type: int

        :Return:错误码,读取的返回值。 type: int32,float

        '''
        value = ctypes.c_float()
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_GetParam(self.handle, _str, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetAccel(self, iaxis):
        '''
        :Description:读取加速度。

        :param sParam:轴号。 type: int

        :Return:错误码,加速度返回值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetAccel(self.handle, iaxis, ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetAddax(self, iaxis):
        '''
        :Description:读取叠加轴。

        :param iaxis:轴号。         type:int

        :Return:错误码,读取的轴叠加轴号。  type: int32,float

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAddax(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAlmIn(self, iaxis, iValue):
        '''
        :Description:设置轴告警信号。

        :param iaxis:轴号。 type: int

        :param iValue:报警信号输入口编号，取消时设定-1   type: int

        :Return:错误码。 type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetAlmIn(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetAlmIn(self, iaxis):
        '''
        :Description:读取告警信号。

        :param iaxis:轴号。         type:int

        :Return:错误码,报警信号输入口返回值。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAlmIn(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAtype(self, iaxis, iValue):
        '''
        :Description:设置轴类型。

        :param iaxis:轴号。 type: int

        :param iValue:轴类型。   type: int

        :Return:错误码。 type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetAtype(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetAtype(self, iaxis):
        '''
        :Description:读取轴类型。

        :param iaxis:轴号。         type:int

        :Return:错误码,轴类型返回值。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAtype(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetAxisStatus(self, iaxis):
        '''
        :Description:读取轴状态。

        :param iaxis:轴号。         type:int

        :Return:错误码,轴状态返回值。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAxisStatus(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAxisAddress(self, iaxis, piValue):
        '''
        :Description:设置轴地址。

        :param iaxis:轴号。 type: int

        :param piValue:轴地址设定值。   type: int

        :Return:错误码。 type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetAxisAddress(self.handle, ctypes.c_int(iaxis), ctypes.c_int(piValue))
        return ret

    def ZAux_Direct_GetAxisAddress(self, iaxis):
        '''
        :Description:读取轴地址。

        :param iaxis:轴号。         type:int

        :Return:错误码,轴地址返回值。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAxisAddress(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAxisEnable(self, iaxis, iValue):
        '''
        :Description:设置轴使能 （只针对总线控制器轴使用有效）。

        :param iaxis:轴号。 type: int

        :param iValue:状态 0-关闭 1- 打开。   type: int

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetAxisEnable(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetAxisEnable(self, iaxis):
        '''
        :Description:读取轴使能状态。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的使能状态。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetAxisEnable(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetClutchRate(self, iaxis, fValue):
        '''
        :Description:设置链接速率。

        :param iaxis:轴号。 type: int

        :param fValue: 同步连接速率。   type: float

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetClutchRate(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetClutchRate(self, iaxis):
        '''
        :Description:读取链接速率。

        :param iaxis:轴号。         type:int

        :Return:错误码,连接速率返回值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetClutchRate(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetCloseWin(self, iaxis, fValue):
        '''
        :Description:设置锁存触发的结束坐标范围点。

        :param iaxis:轴号。 type: int

        :param fValue: 设定的范围值。   type: float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetCloseWin(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetCloseWin(self, iaxis):
        '''
        :Description:读取锁存触发的结束坐标范围点。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的范围值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetCloseWin(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetCornerMode(self, iaxis, iValue):
        '''
        :Description:设置拐角减速。

        :param iaxis:轴号。         type:int

        :param iValue: 拐角减速模式。  type:int

        :Return:错误码。  type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetCornerMode(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetCornerMode(self, iaxis):
        '''
        :Description:读取拐角减速。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的拐角模式。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetCornerMode(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetCreep(self, iaxis, fValue):
        '''
        :Description:设置回零爬行速度。

        :param iaxis:轴号。         type:int

        :param fValue: 设置的速度值。  type:float

        :Return:错误码。  type: int32

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_SetCreep(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue),
                                           ctypes.byref(value))
        return ret

    def ZAux_Direct_GetCreep(self, iaxis):
        '''
        :Description:读取回零爬行速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的爬行速度值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetCreep(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDatumIn(self, iaxis, iValue):
        '''
        :Description:设置原点信号   设定-1为取消原点设置。

        :param iaxis:轴号。         type:int

        :param iValue: 设置的原点信号输入口编号。  type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetDatumIn(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetDatumIn(self, iaxis):
        '''
        :Description:读取原点信号。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回原点输入口编号。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetDatumIn(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDecel(self, iaxis, fValue):
        '''
        :Description:设置减速度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的减速度值。  type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetDecel(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetDecel(self, iaxis):
        '''
        :Description:读取减速度

        :param iaxis:轴号。         type:int

        :Return:错误码,设定的减速度返回值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDecel(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDecelAngle(self, iaxis, fValue):
        '''
        :Description:设置拐角减速角度，开始减速角度，单位为弧度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的拐角减速角度。  type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetDecelAngle(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetDecelAngle(self, iaxis):
        '''
        :Description:读取拐角开始减速角度，单位为弧度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的拐角减速角度。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDecelAngle(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDpos(self, iaxis, fValue):
        '''
        :Description:设置轴位置。

        :param iaxis:轴号。         type:int

        :param fValue:设置的坐标值。  type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetDpos(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetDpos(self, iaxis):
        '''
        :Description:读取轴位置。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的命令位置坐标。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDpos(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetEncoder(self, iaxis):
        '''
        :Description:读取内部编码器值  （总线绝对值伺服时为绝对值位置）。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的内部编码器值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetEncoder(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetEndMove(self, iaxis):
        '''
        :Description:读取当前运动的最终位置。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的最终位置。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetEndMove(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetEndMoveBuffer(self, iaxis):
        '''
        :Description:读取当前和缓冲中运动的最终位置，可以用于相对绝对转换。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的最终位置。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetEndMoveBuffer(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetEndMoveSpeed(self, iaxis, fValue):
        '''
        :Description:设置SP运动的结束速度。

        :param iaxis:轴号。         type:int

        :param fValue:设定的速度值。  type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetEndMoveSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetEndMoveSpeed(self, iaxis):
        '''
        :Description:读取SP运动的结束速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的速度值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetEndMoveSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetErrormask(self, iaxis, iValue):
        '''
        :Description:设置错误标记,和AXISSTATUS做与运算来决定哪些错误需要关闭WDOG。

        :param iaxis:轴号。         type:int

        :param iValue:设置值。         type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetErrormask(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetErrormask(self, iaxis):
        '''
        :Description:读取错误标记,和AXISSTATUS做与运算来决定哪些错误需要关闭WDOG。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的标记值。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetErrormask(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFastJog(self, iaxis, iValue):
        '''
        :Description:设置快速JOG输入。

        :param iaxis:轴号。         type:int

        :param iValue:快速JOG输入口编号。         type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFastJog(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetFastJog(self, iaxis):
        '''
        :Description:读取快速JOG输入。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的JOG输入口编号。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetFastJog(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFastDec(self, iaxis, fValue):
        '''
        :Description:设置快速减速度。

        :param iaxis:轴号。         type:int

        :param fValue:设定的快速减速度。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFastDec(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetFastDec(self, iaxis):
        '''
        :Description:读取快速减速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的快速减速度。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFastDec(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetFe(self, iaxis):
        '''
        :Description:读取随动误差。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的随动误差。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFe(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFeLimit(self, iaxis, fValue):
        '''
        :Description:设置最大允许的随动误差值。

        :param iaxis:轴号。         type:int

        :param fValue:设置的最大误差值。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFeLimit(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetFeLimit(self, iaxis):
        '''
        :Description:读取最大允许的随动误差值。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的设置最大误差值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFeLimit(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFRange(self, iaxis, fValue):
        '''
        :Description:设置报警时随动误差值。

        :param iaxis:轴号。         type:int

        :param fValue:设置的误差值。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFRange(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetFeRange(self, iaxis):
        '''
        :Description:读取报警时的随动误差值。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的报警误差值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFeRange(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFholdIn(self, iaxis, iValue):
        '''
        :Description:设置保持输入。

        :param iaxis:轴号。         type:int

        :param fValue:设置的输入口编号。      type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFholdIn(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetFholdIn(self, iaxis):
        '''
        :Description:读取保持输入。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回输入HOLDIN输入口编号。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetFholdIn(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFhspeed(self, iaxis, pfValue):
        '''
        :Description:设置轴保持速度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的速度值。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFhspeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(pfValue))
        return ret

    def ZAux_Direct_GetFhspeed(self, iaxis):
        '''
        :Description:读取轴保持速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的保持速度。  type: int32,int

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFhspeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetForceSpeed(self, iaxis, fValue):
        '''
        :Description:设置SP运动的运行速度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的速度值。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetForceSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetForceSpeed(self, iaxis):
        '''
        :Description:读取SP运动的运行速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回SP运动速度值。  type: int32,int

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetForceSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFsLimit(self, iaxis, fValue):
        '''
        :Description:设置正向软限位,取消时设置一个较大值即可。

        :param iaxis:轴号。         type:int

        :param fValue:设定的限位值。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFsLimit(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetFsLimit(self, iaxis):
        '''
        :Description:读取正向软限位。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的正向限位坐标。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFsLimit(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFullSpRadius(self, iaxis, fValue):
        '''
        :Description:设置小圆限速最小半径。

        :param iaxis:轴号。         type:int

        :param fValue:设置的最小半径。      type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFullSpRadius(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetFullSpRadius(self, iaxis):
        '''
        :Description:读取小圆限速最小半径。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的限速半径。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetFullSpRadius(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFwdIn(self, iaxis, iValue):
        '''
        :Description:设置正向硬限位输入  设置成-1时表示不设置限位。

        :param iaxis:轴号。         type:int

        :param iValue:设置的限位输入口编号。      type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetFwdIn(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetFwdIn(self, iaxis):
        '''
        :Description:读取正向硬限位输入。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回正向限位输入口编号。  type: int32,float

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetFwdIn(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetFwdJog(self, iaxis, iValue):
        '''
        :Description:设置正向JOG输入。

        :param iaxis:轴号。         type:int

        :param iValue:设置的JOG输入口编号。      type:int

        :Return:错误码。  type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetFwdJog(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetFwdJog(self, iaxis):
        '''
        :Description:读取正向JOG输入。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的JOG输入口编号。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetFwdJog(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetIfIdle(self, iaxis):
        '''
        :Description:读取轴是否运动结束。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回运行状态 0-运动中 -1 停止。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetIfIdle(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetInvertStep(self, iaxis, iValue):
        '''
        :Description:设置脉冲输出模式。

        :param iaxis:轴号。         type:int

        :param iValue:设定的脉冲输出模式 脉冲+方向/双脉冲。         type:int

        :Return:错误码。  type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetInvertStep(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetInvertStep(self, iaxis):
        '''
        :Description:读取脉冲输出模式。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的脉冲模式。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetInvertStep(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetInterpFactor(self, iaxis, iValue):
        '''
        :Description:设置插补时轴是否参与速度计算,缺省参与(1)。此参数只对直线和螺旋的第三个轴起作用。

        :param iaxis:轴号。         type:int

        :param iValue:模式 0-不参数 1-参与。         type:int

        :Return:错误码。  type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetInterpFactor(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetInterpFactor(self, iaxis):
        '''
        :Description:读取插补时轴是否参与速度计算，缺省参与(1)。此参数只对直线和螺旋的第三个轴起作用。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的速度计算模式。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetInterpFactor(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetJogSpeed(self, iaxis, fValue):
        '''
        :Description:设置JOG时速度。

        :param iaxis:轴号。         type:int

        :param fValue:设定的速度值。         type:float

        :Return:错误码。  type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetJogSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetJogSpeed(self, iaxis):
        '''
        :Description:读取插补时轴是否参与速度计算，缺省参与(1)。此参数只对直线和螺旋的第三个轴起作用。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的速度计算模式。  type: int32,int

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetJogSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetLoaded(self, iaxis):
        '''
        :Description:读取当前除了当前运动是否还有缓冲。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回状态值  -1 没有剩余函数 0-还有剩余运动。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetLoaded(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetLinkax(self, iaxis):
        '''
        :Description:读取当前链接运动的参考轴号。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回链接的参考轴号。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetLinkax(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetLspeed(self, iaxis, fValue):
        '''
        :Description:设置轴起始速度。

        :param iaxis:轴号。         type:int

        :param fValue:设定的速度值。         type:float

        :Return:错误码。  type: int32
        '''

        ret = zauxdll.ZAux_Direct_SetLspeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetLspeed(self, iaxis):
        '''
        :Description:读取轴起始速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的起始速度值。  type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetLspeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetHomeWait(self, iaxis, fValue):
        '''
        :Description:设置回零反找等待时间。

        :param iaxis:轴号。         type:int

        :param fValue:回零反找等待时间 MS。         type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetHomeWait(self.handle, ctypes.c_int(iaxis), ctypes.c_int(fValue))
        return ret

    def ZAux_Direct_GetHomeWait(self, iaxis):
        '''
        :Description:读取回零反找等待时间。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的反找等待时间。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetHomeWait(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMark(self, iaxis):
        '''
        :Description:读取编码器锁存示教返回状态。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的锁存触发状态 -1-锁存触发 0-未触发。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMark(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMarkB(self, iaxis):
        '''
        :Description:读取编码器锁存b返回状态。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的锁存触发状态 -1-锁存触发 0-未触发。  type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMarkB(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetMaxSpeed(self, iaxis, iValue):
        '''
        :Description:设置脉冲输出最高频率。

        :param iaxis:轴号。         type:int

        :param iValue:设置的最高脉冲频率    type:int

        :Return:错误码。  type: int32,int

        '''

        ret = zauxdll.ZAux_Direct_SetMaxSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetMaxSpeed(self, iaxis):
        '''
        :Description:读取脉冲输出最高频率。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的脉冲频率。    type:int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMaxSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetMerge(self, iaxis, iValue):
        '''
        :Description:设置连续插补。

        :param iaxis:轴号。         type:int

        :param iValue:连续插补开关 0-关闭连续插补 1-打开连续插补。 type:int

        :Return:错误码,返回的反找等待时间。  type: int32,int

        '''
        ret = zauxdll.ZAux_Direct_SetMerge(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetMerge(self, iaxis):
        '''
        :Description:读取连续插补状态。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的连续插补开关状态。    type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMerge(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMovesBuffered(self, iaxis):
        '''
        :Description:读取当前被缓冲起来的运动个数。

        :param iaxis:轴号。         type:int

        :Return:错误码,缓冲运动数。    type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMovesBuffered(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMoveCurmark(self, iaxis):
        '''
        :Description:读取当前正在运动指令的MOVE_MARK标号。

        :param iaxis:轴号。         type:int

        :Return:错误码,当前MARK标号。    type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMoveCurmark(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetMovemark(self, iaxis, iValue):
        '''
        :Description:设置运动指令的MOVE_MARK标号 每当有运动进入轴运动缓冲时MARK自动+1。

        :param iaxis:轴号。         type:int

        :param iValue:设定的MARK值。 type:int

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetMovemark(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetMpos(self, iaxis, fValue):
        '''
        :Description:设置反馈位置。

        :param iaxis:轴号。         type:int

        :param fValue:设置的反馈位置。 type:float

        :Return:错误码。  type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetMpos(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetMpos(self, iaxis):
        '''
        :Description:读取反馈位置。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的轴反馈位置坐标。    type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetMpos(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMspeed(self, iaxis):
        '''
        :Description:读取反馈速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的编码器反馈速度。   type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetMspeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetMtype(self, iaxis):
        '''
        :Description:读取当前正在运动指令类型。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回当前的运动类型。    type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetMtype(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetOffpos(self, iaxis, fValue):
        '''
        :Description:设置修改偏移位置。

        :param iaxis:轴号。         type:int

        :param fValue:设置的反馈位置。 type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetOffpos(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetOffpos(self, iaxis):
        '''
        :Description:读取修改偏移位置。
        :param iaxis:轴号。         type:int

        :Return:错误码,返回的偏移坐标值    type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetOffpos(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetOpenWin(self, iaxis, fValue):
        '''
        :Description:设置锁存触发的结束坐标范围点。

        :param iaxis:轴号。         type:int

        :param fValue:设置的坐标值。 type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetOpenWin(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetOpenWin(self, iaxis):
        '''
        :Description:读取锁存触发的结束坐标范围点。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的结束坐标值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetOpenWin(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetRegPos(self, iaxis):
        '''
        :Description:读取返回锁存的测量反馈位置(MPOS)。

        :param iaxis:轴号。         type:int

        :Return:错误码,锁存的坐标位置。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetRegPos(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetRegPosB(self, iaxis):
        '''
        :Description:读取返回锁存的测量反馈位置(MPOS)。

        :param iaxis:轴号。         type:int

        :Return:错误码,锁存的坐标位置。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetRegPosB(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetRemain(self, iaxis):
        '''
        :Description:读取返回轴当前运动还未完成的距离。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的剩余距离。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetRemain(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetRemain_LineBuffer(self, iaxis):
        '''
        :Description:参数  轴剩余的缓冲, 按直线段来计算。

        :param iaxis:轴号。         type:int

        :Return:错误码,剩余的直线缓冲数量。 type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetRemain_LineBuffer(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetRemain_Buffer(self, iaxis):
        '''
        :Description:参数  轴剩余的缓冲, 按最复杂的空间圆弧来计算。

        :param iaxis:轴号。         type:int

        :Return:错误码,剩余的缓冲数量。 type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetRemain_Buffer(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetRepDist(self, iaxis, fValue):
        '''
        :Description:设置锁存触发的结束坐标范围点。

        :param iaxis:轴号。         type:int

        :param fValue:设置的坐标值。 type:float

        :Return:错误码。  type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetRepDist(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetRepDist(self, iaxis):
        '''
        :Description:读取根据REP_OPTION设置来自动循环轴DPOS和MPOS坐标。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的循环坐标值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetRepDist(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetRepOption(self, iaxis, iValue):
        '''
        :Description:设置坐标重复设置。

        :param iaxis:轴号。         type:int

        :param iValue:模式。 type:int

        :Return:错误码。  type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetRepOption(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetRepOption(self, iaxis):
        '''
        :Description:读取坐标重复设置。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的模式。 type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetRepOption(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetRevIn(self, iaxis, iValue):
        '''
        :Description:设置负向硬件限位开关对应的输入点编号，-1无效。

        :param iaxis:轴号。         type:int

        :param iValue:设置的输入口编号。      type:int

        :Return:错误码。          type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetRevIn(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetRevIn(self, iaxis):
        '''
        :Description:读取负向硬件限位开关对应的输入点编号，-1无效。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的负向限位输入口编号。 type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetRevIn(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetRevJog(self, iaxis, iValue):
        '''
        :Description:设置负向JOG输入对应的输入点编号,-1无效。

        :param iaxis:轴号。         type:int

        :param iValue:设置的输入口编号。         type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetRevJog(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_GetRevJog(self, iaxis):
        '''
        :Description:读取负向JOG输入对应的输入点编号,-1无效。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的输入口编号。 type: int32,int

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetRevJog(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetRsLimit(self, iaxis, fValue):
        '''
        :Description:设置负向软限位位置。  设置一个较大的值时认为取消限位。

        :param iaxis:轴号。         type:int

        :param fValue:负向限位值。         type:float

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetRsLimit(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetRsLimit(self, iaxis):
        '''
        :Description:读取负向软限位位置。

        :param iaxis:轴号。         type:int

        :Return:错误码,设定的限位值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetRsLimit(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetSpeed(self, iaxis, fValue):
        '''
        :Description:设置轴速度,单位为units/s,当多轴运动时,作为插补运动的速度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的速度值。         type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetSpeed(self, iaxis):
        '''
        :Description:读取轴速度,单位为units/s,当多轴运动时,作为插补运动的速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的速度值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetSramp(self, iaxis, fValue):
        '''
        :Description:设置 S曲线设置。 0-梯形加减速。

        :param iaxis:轴号。         type:int

        :param fValue:S曲线平滑时间MS。         type:float

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_SetSramp(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetSramp(self, iaxis):
        '''
        :Description:读取 S曲线设置。

        :param iaxis:轴号。         type:int

        :Return:错误码,平滑时间。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetSramp(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetStartMoveSpeed(self, iaxis, fValue):
        '''
        :Description:设置 自定义速度的SP运动的起始速度。

        :param iaxis:轴号。         type:int

        :param fValue:设置的速度值。         type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetStartMoveSpeed(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetStartMoveSpeed(self, iaxis):
        '''
        :Description:读取自定义速度的SP运动的起始速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的SP运动起始速度值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetStartMoveSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetStopAngle(self, iaxis, fValue):
        '''
        :Description:设置 减速到最低的最小拐角 弧度制。

        :param iaxis:轴号。         type:int

        :param fValue:设置的角度值。         type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetStopAngle(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetStopAngle(self, iaxis):
        '''
        :Description:取减速到最低的最小拐角 弧度制。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的拐角停止角度。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetStopAngle(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetZsmooth(self, iaxis, fValue):
        '''
        :Description:设置 减速倒角半径。

        :param iaxis:轴号。         type:int

        :param fValue:倒角半径。         type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetZsmooth(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetZsmooth(self, iaxis):
        '''
        :Description:读取倒角半径。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的倒角半径值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetZsmooth(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetUnits(self, iaxis, fValue):
        '''
        :Description:设置 脉冲当量。

        :param iaxis:轴号。         type:int

        :param fValue:设置的当量值。         type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_SetUnits(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetUnits(self, iaxis):
        '''
        :Description:读取脉冲当量。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的脉冲当量。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetUnits(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetVectorBuffered(self, iaxis):
        '''
        :Description:读取返回轴当前当前运动和缓冲运动还未完成的距离。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的剩余的距离。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetVectorBuffered(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetVpSpeed(self, iaxis):
        '''
        :Description:读取当前轴运行的命令速度。

        :param iaxis:轴号。         type:int

        :Return:错误码,返回的当前速度值。 type: int32,float

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetVpSpeed(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetVariablef(self, pname):
        '''
        :Description:全局变量读取, 也可以是参数等等。

        :param iaxis:轴号。         type:int

        :param pname:全局变量名称/或者指定轴号的轴参数名称DPOS(0)。  type:string

        :Return:错误码,返回值。 type: int32,float

        '''
        _str = pname.encode('utf-8')
        value = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetVariablef(self.handle, _str, ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetVariableInt(self, pname):
        '''
        :Description:全局变量读取, 也可以是参数等等。

        :param iaxis:轴号。         type:int

        :param pname:全局变量名称/或者指定轴号的轴参数名称DPOS(0)。  type:string

        :Return:错误码,返回值。 type: int32,int

        '''
        _str = pname.encode('utf-8')
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetVariableInt(self.handle, _str, ctypes.byref(value))
        return ret, value

    ##############下面的运动函数支持直接调用，并不是所有的指令都支持,必须 20130901 以后的控制器版本支持###############

    def ZAux_Direct_Base(self, imaxaxises, piAxislist):
        '''
        :Description:BASE指令调用
         仅仅修改在线命令的BASE列表,不对控制器的运行任务的BASE进行修改.
         修改后,后续的所有MOVE等指令都是以这个BASE为基础  。

        :param imaxaxises:参与轴数。  type:int

        :param piAxislist:轴列表。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_Base(self.handle, ctypes.c_int(imaxaxises), Axislistarray)
        return ret

    def ZAux_Direct_Defpos(self, iaxis, pfDpos):
        '''
        :Description:定义DPOS,不建议使用,可以直接调用SETDPOS达到同样效果。

        :param iaxis:轴号。         type:int

        :param pfDpos:设置的坐标值。  type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Defpos(self.handle, ctypes.c_int(iaxis), ctypes.c_float(pfDpos))
        return ret

    def ZAux_Direct_Move(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:多轴相对直线插补  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。  type:int

        :param piAxislist:轴列表。  type:int

        :param pfDisancelist:距离列表。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_MoveSp(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:相对多轴直线插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。  type:int

        :param piAxislist:轴列表。  type:int

        :param pfDisancelist:距离列表。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_MoveAbs(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:绝对多轴直线插补  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。  type:int

        :param piAxislist:轴列表。  type:int

        :param pfDisancelist:距离列表。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_MoveAbsSp(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:绝对多轴直线插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。  type:int

        :param piAxislist:轴列表。  type:int

        :param pfDisancelist:距离列表。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_MoveModify(self, iaxis, pfDisance):
        '''
        :Description:运动中修改结束位置  20130901 以后的控制器版本支持。

        :param iaxis:轴号。  type:int

        :param pfDisance:绝对距离。  type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveModify(self.handle, ctypes.c_int(iaxis), ctypes.c_float(pfDisance))
        return ret

    def ZAux_Direct_MoveCirc(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection):
        '''
        :Description:相对圆心定圆弧插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2:第二个轴运动圆心，相对与起始点。  type:float

        :param idirection:0-逆时针,1-顺时针。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCirc(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(fend1),
                                           ctypes.c_float(fend2), ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                           ctypes.c_int(idirection))
        return ret

    def ZAux_Direct_MoveCircSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection):
        '''
        :Description:相对圆心定圆弧插补运动 插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2:第二个轴运动圆心，相对与起始点。  type:float

        :param idirection:0-逆时针,1-顺时针。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCircSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                             ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                             ctypes.c_float(fcenter2), ctypes.c_int(idirection))
        return ret

    def ZAux_Direct_MoveCircAbs(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection):
        '''
        :Description:绝对圆心圆弧插补运动  20130901 以后的控制器版本支持  无法画整圆。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2:第二个轴运动圆心，相对与起始点。  type:float

        :param idirection:0-逆时针,1-顺时针。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCircAbs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                              ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                              ctypes.c_float(fcenter2), ctypes.c_int(idirection))
        return ret

    def ZAux_Direct_MoveCircAbsSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection):
        '''
        :Description:绝对圆心圆弧插补运动  20130901 以后的控制器版本支持  无法画整圆。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2:第二个轴运动圆心，相对与起始点。  type:float

        :param idirection:0-逆时针,1-顺时针。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCircAbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                                ctypes.c_float(fcenter2), ctypes.c_int(idirection))
        return ret

    def ZAux_Direct_MoveCirc2(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2):
        '''
        :Description:相对3点定圆弧插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点，相对起始点距离。  type:float

        :param fmid2: 第二个轴中间点，相对起始点距离。  type:float

        :param fend1:第一个轴结束点，相对起始点距离。  type:float

        :param fend2: 第二个轴结束点，相对起始点距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCirc2(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(fmid1),
                                            ctypes.c_float(fmid2), ctypes.c_float(fend1), ctypes.c_float(fend2))
        return ret

    def ZAux_Direct_MoveCirc2Abs(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2):
        '''
        :Description:绝对3点定圆弧插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点，相对起始点距离。  type:float

        :param fmid2: 第二个轴中间点，相对起始点距离。  type:float

        :param fend1:第一个轴结束点，相对起始点距离。  type:float

        :param fend2: 第二个轴结束点，相对起始点距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCirc2Abs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                               ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                               ctypes.c_float(fend2))
        return ret

    def ZAux_Direct_MoveCirc2Sp(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2):
        '''
        :Description:相对3点定圆弧插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点，相对起始点距离。  type:float

        :param fmid2: 第二个轴中间点，相对起始点距离。  type:float

        :param fend1:第一个轴结束点，相对起始点距离。  type:float

        :param fend2: 第二个轴结束点，相对起始点距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCirc2Sp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                              ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                              ctypes.c_float(fend2))
        return ret

    def ZAux_Direct_MoveCirc2AbsSp(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2):
        '''
        :Description:绝对3点定圆弧插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点，相对起始点距离。  type:float

        :param fmid2: 第二个轴中间点，相对起始点距离。  type:float

        :param fend1:第一个轴结束点，相对起始点距离。  type:float

        :param fend2: 第二个轴结束点，相对起始点距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveCirc2AbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                 ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                                 ctypes.c_float(fend2))
        return ret

    def ZAux_Direct_MHelical(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fDistance3,
                             imode):
        '''
        :Description:相对3轴圆心螺旋插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2: 第二个轴运动圆心，相对与起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode:  第三轴的速度计算:0(缺省)第三轴参与速度计算。1第三轴不参与速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelical(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(fend1),
                                           ctypes.c_float(fend2), ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                           ctypes.c_float(idirection), ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelicalAbs(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fDistance3,
                                imode):
        '''
        :Description:绝对3轴圆心螺旋插补运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2: 第二个轴运动圆心，相对与起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode:  第三轴的速度计算:0(缺省)第三轴参与速度计算。1第三轴不参与速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelicalAbs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                              ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                              ctypes.c_float(fcenter2), ctypes.c_float(idirection),
                                              ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelicalSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fDistance3,
                               imode):
        '''
        :Description:相对3轴圆心螺旋插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2: 第二个轴运动圆心，相对与起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode:  第三轴的速度计算:0(缺省)第三轴参与速度计算。1第三轴不参与速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelicalSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                             ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                             ctypes.c_float(fcenter2), ctypes.c_float(idirection),
                                             ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelicalAbsSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection,
                                  fDistance3, imode):
        '''
        :Description:绝对3轴圆心螺旋插补运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第一个轴运动坐标。  type:float

        :param fend2:第二个轴运动坐标。  type:float

        :param fcenter1:第一个轴运动圆心，相对与起始点。  type:float

        :param fcenter2: 第二个轴运动圆心，相对与起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode:  第三轴的速度计算:0(缺省)第三轴参与速度计算。1第三轴不参与速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelicalAbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                                ctypes.c_float(fcenter2), ctypes.c_float(idirection),
                                                ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelical2(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2, fDistance3, imode):
        '''
        :Description:相对3轴 3点画螺旋插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点。  type:float

        :param fmid2:第二个轴中间点。  type:float

        :param fend1:第一个轴结束点。  type:float

        :param fend2: 第二个轴结束点。  type:float

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode: 第三轴的速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelical2(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(fmid1),
                                            ctypes.c_float(fmid2), ctypes.c_float(fend1), ctypes.c_float(fend2),
                                            ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelical2Abs(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2, fDistance3, imode):
        '''
        :Description:绝对3轴 3点画螺旋插补运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点。  type:float

        :param fmid2:第二个轴中间点。  type:float

        :param fend1:第一个轴结束点。  type:float

        :param fend2: 第二个轴结束点。  type:float

        :param fDistance3: 第三个轴运动结束点。  type:float

        :param imode: 第三轴的速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelical2Abs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                               ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                               ctypes.c_float(fend2), ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelical2Sp(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2, fDistance3, imode):
        '''
        :Description:相对3轴 3点画螺旋插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点。  type:float

        :param fmid2:第二个轴中间点。  type:float

        :param fend1:第一个轴结束点。  type:float

        :param fend2: 第二个轴结束点。  type:float

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode: 第三轴的速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelical2Sp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                              ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                              ctypes.c_float(fend2), ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MHelical2AbsSp(self, imaxaxises, piAxislist, fmid1, fmid2, fend1, fend2, fDistance3, imode):
        '''
        :Description:绝对3轴 3点画螺旋插补SP运动  20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fmid1:第一个轴中间点。  type:float

        :param fmid2:第二个轴中间点。  type:float

        :param fend1:第一个轴结束点。  type:float

        :param fend2: 第二个轴结束点。  type:float

        :param fDistance3: 第三个轴运动距离。  type:float

        :param imode: 第三轴的速度计算。  type:int

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MHelical2AbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                 ctypes.c_float(fmid1), ctypes.c_float(fmid2), ctypes.c_float(fend1),
                                                 ctypes.c_float(fend2), ctypes.c_float(fDistance3), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MEclipse(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis, fBDis):
        '''
        :Description:相对椭圆插补 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipse(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(fend1),
                                           ctypes.c_float(fend2), ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                           ctypes.c_int(idirection), ctypes.c_float(fADis), ctypes.c_float(fBDis))
        return ret

    def ZAux_Direct_MEclipseAbs(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis,
                                fBDis):
        '''
        :Description:绝对椭圆插补 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseAbs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                              ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                              ctypes.c_float(fcenter2), ctypes.c_int(idirection), ctypes.c_float(fADis),
                                              ctypes.c_float(fBDis))
        return ret

    def ZAux_Direct_MEclipseSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis,
                               fBDis):
        '''
        :Description:相对椭圆插补SP运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                             ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                             ctypes.c_float(fcenter2), ctypes.c_int(idirection), ctypes.c_float(fADis),
                                             ctypes.c_float(fBDis))
        return ret

    def ZAux_Direct_MEclipseAbsSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis,
                                  fBDis):
        '''
        :Description:绝对椭圆插补SP运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseAbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fcenter1),
                                                ctypes.c_float(fcenter2), ctypes.c_int(idirection),
                                                ctypes.c_float(fADis), ctypes.c_float(fBDis))
        return ret

    def ZAux_Direct_MEclipseHelical(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis,
                                    fBDis, fDistance3):
        '''
        :Description:相对 椭圆 + 螺旋插补运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :param fDistance3: 第三个轴的运动距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseHelical(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                  ctypes.c_float(fend1), ctypes.c_float(fend2),
                                                  ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                                  ctypes.c_int(idirection), ctypes.c_float(fADis),
                                                  ctypes.c_float(fBDis), ctypes.c_float(fDistance3))
        return ret

    def ZAux_Direct_MEclipseHelicalAbs(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection,
                                       fADis, fBDis, fDistance3, ):
        '''
        :Description:绝对椭圆 + 螺旋插补运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :param fDistance3: 第三个轴的运动距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseHelicalAbs(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                     ctypes.c_float(fend1), ctypes.c_float(fend2),
                                                     ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                                     ctypes.c_int(idirection), ctypes.c_float(fADis),
                                                     ctypes.c_float(fBDis), ctypes.c_float(fDistance3))
        return ret

    def ZAux_Direct_MEclipseHelicalSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection, fADis,
                                      fBDis, fDistance3):
        '''
        :Description:相对 椭圆 + 螺旋插补SP运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :param fDistance3: 第三个轴的运动距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseHelicalSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                    ctypes.c_float(fend1), ctypes.c_float(fend2),
                                                    ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                                    ctypes.c_int(idirection), ctypes.c_float(fADis),
                                                    ctypes.c_float(fBDis), ctypes.c_float(fDistance3))
        return ret

    def ZAux_Direct_MEclipseHelicalAbsSp(self, imaxaxises, piAxislist, fend1, fend2, fcenter1, fcenter2, idirection,
                                         fADis, fBDis, fDistance3):
        '''
        :Description:绝对椭圆 + 螺旋插补SP运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:终点第一个轴运动坐标，相对于起始点。  type:float

        :param fend2:终点第二个轴运动坐标，相对于起始点。  type:float

        :param fcenter1: 中心第一个轴运动坐标，相对于起始点。  type:float

        :param fcenter2: 中心第二个轴运动坐标，相对于起始点。  type:float

        :param idirection: 0-逆时针,1-顺时针。  type:int

        :param fADis: 第一轴的椭圆半径，半长轴或者半短轴都可。  type:float

        :param fBDis: 第二轴的椭圆半径,半长轴或者半短轴都可,AB相等时自动为圆弧或螺。  type:float

        :param fDistance3: 第三个轴的运动距离。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MEclipseHelicalAbsSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                                       ctypes.c_float(fend1), ctypes.c_float(fend2),
                                                       ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                                       ctypes.c_int(idirection), ctypes.c_float(fADis),
                                                       ctypes.c_float(fBDis), ctypes.c_float(fDistance3))
        return ret

    def ZAux_Direct_MSpherical(self, imaxaxises, piAxislist, fend1, fend2, fend3, fcenter1, fcenter2, fcenter3, imode,
                               fcenter4, fcenter5):
        '''
        :Description:空间圆弧 + 螺旋插补运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第1个轴运动距离参数1	相对与起点。  type:float

        :param fend2:第2个轴运动距离参数1	相对与起点。  type:float

        :param fend3:第3个轴运动距离参数1	相对与起点。  type:float

        :param fcenter1: 第1个轴运动距离参数2	相对与起点。  type:float

        :param fcenter2: 第2个轴运动距离参数2	相对与起点。  type:float

        :param fcenter3: 第3个轴运动距离参数2   相对与起点。  type:float

        :param imode: 指定前面参数的意义。  type:int
                      0 当前点,中间点,终点三点定圆弧,距离参数1为终点距离,距离参数2为中间点距离。
                      1 走最小的圆弧,距离参数1为终点距离,距离参数2为圆心的距离。
                      2 当前点,中间点,终点三点定圆,距离参数1为终点距离,距离参数2为中间点距离。
                      3 先走最小的圆弧,再继续走完整圆,距离参数1为终点距离,离参数2为圆心的距离。

        :param fcenter4: 第4个轴运动距离参数 。  type:float

        :param fcenter5: 第5个轴运动距离参数 。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MSpherical(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                             ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fend3),
                                             ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                             ctypes.c_float(fcenter3), ctypes.c_int(imode), ctypes.c_float(fcenter4),
                                             ctypes.c_float(fcenter5))
        return ret

    def ZAux_Direct_MSphericalSp(self, imaxaxises, piAxislist, fend1, fend2, fend3, fcenter1, fcenter2, fcenter3, imode,
                                 fcenter4, fcenter5):
        '''
        :Description:空间圆弧 + 螺旋 插补SP运动 20130901 以后的控制器版本支持。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param fend1:第1个轴运动距离参数1	相对与起点。  type:float

        :param fend2:第2个轴运动距离参数1	相对与起点。  type:float

        :param fend3:第3个轴运动距离参数1	相对与起点。  type:float

        :param fcenter1: 第1个轴运动距离参数2	相对与起点。  type:float

        :param fcenter2: 第2个轴运动距离参数2	相对与起点。  type:float

        :param fcenter3: 第3个轴运动距离参数2   相对与起点。  type:float

        :param imode: 指定前面参数的意义。  type:int
                      0 当前点,中间点,终点三点定圆弧,距离参数1为终点距离,距离参数2为中间点距离。
                      1 走最小的圆弧,距离参数1为终点距离,距离参数2为圆心的距离。
                      2 当前点,中间点,终点三点定圆,距离参数1为终点距离,距离参数2为中间点距离。
                      3 先走最小的圆弧,再继续走完整圆,距离参数1为终点距离,离参数2为圆心的距离。

        :param fcenter4: 第4个轴运动距离参数 。  type:float

        :param fcenter5: 第5个轴运动距离参数 。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MSphericalSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                               ctypes.c_float(fend1), ctypes.c_float(fend2), ctypes.c_float(fend3),
                                               ctypes.c_float(fcenter1), ctypes.c_float(fcenter2),
                                               ctypes.c_float(fcenter3), ctypes.c_int(imode), ctypes.c_float(fcenter4),
                                               ctypes.c_float(fcenter5))
        return ret

    def ZAux_Direct_MoveSpiral(self, imaxaxises, piAxislist, centre1, centre2, circles, pitch, distance3, distance4):
        '''
        :Description:渐开线圆弧插补运动,相对移动方式,当起始半径0直接扩散时从0角度开始。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param centre1:第1轴圆心的相对距离。  type:float

        :param centre2:第2轴圆心的相对距离。  type:float

        :param circles:要旋转的圈数，可以为小数圈，负数表示顺时针。  type:float

        :param pitch: 每圈的扩散距离，可以为负。  type:float

        :param distance3: 第3轴螺旋的功能,指定第3轴的相对距离,此轴不参与速度计算。  type:float

        :param distance4: 第4轴螺旋的功能,指定第4轴的相对距离,此轴不参与速度计算。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveSpiral(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                             ctypes.c_float(centre1), ctypes.c_float(centre2), ctypes.c_float(circles),
                                             ctypes.c_float(pitch), ctypes.c_float(distance3),
                                             ctypes.c_float(distance4))
        return ret

    def ZAux_Direct_MoveSpiralSp(self, imaxaxises, piAxislist, centre1, centre2, circles, pitch, distance3, distance4):
        '''
        :Description:渐开线圆弧插补SP运动,相对移动方式,当起始半径0直接扩散时从0角度开始。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param centre1:第1轴圆心的相对距离。  type:float

        :param centre2:第2轴圆心的相对距离。  type:float

        :param circles:要旋转的圈数，可以为小数圈，负数表示顺时针。  type:float

        :param pitch: 每圈的扩散距离，可以为负。  type:float

        :param distance3: 第3轴螺旋的功能,指定第3轴的相对距离,此轴不参与速度计算。  type:float

        :param distance4: 第4轴螺旋的功能,指定第4轴的相对距离,此轴不参与速度计算。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveSpiralSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                               ctypes.c_float(centre1), ctypes.c_float(centre2),
                                               ctypes.c_float(circles), ctypes.c_float(pitch),
                                               ctypes.c_float(distance3), ctypes.c_float(distance4))
        return ret

    def ZAux_Direct_MoveSmooth(self, imaxaxises, piAxislist, end1, end2, end3, next1, next2, next3, radius):
        '''
        :Description:空间直线运动，根据下一个直线运动的绝对坐标在拐角自动插入圆弧，加入圆弧后会使得运动的终点与直线的终点不一致，拐角过大时不会插入圆弧，当距离不够时会自动减小半径。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param end1: 第1个轴运动绝对坐标。  type:float

        :param end2: 第2个轴运动绝对坐标。  type:float

        :param end3: 第3个轴运动绝对坐标。  type:float

        :param next1: 第1个轴下一个直线运动绝对坐标。  type:float

        :param next2: 第2个轴下一个直线运动绝对坐标。  type:float

        :param next3: 第3个轴下一个直线运动绝对坐标。  type:float

        :param radius: 插入圆弧的半径，当过大的时候自动缩小。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveSmooth(self.handle, ctypes.c_int(imaxaxises), Axislistarray, ctypes.c_float(end1),
                                             ctypes.c_float(end2), ctypes.c_float(end3), ctypes.c_float(next1),
                                             ctypes.c_float(next2), ctypes.c_float(next3), ctypes.c_float(radius))
        return ret

    def ZAux_Direct_MoveSmoothSp(self, imaxaxises, piAxislist, end1, end2, end3, next1, next2, next3, radius):
        '''
        :Description:空间直线插补SP运动,根据下一个直线运动的绝对坐标在拐角自动插入圆弧,加入圆弧后会使得运动的终点与直线的终点不一致,拐角过大时不会插入圆弧,当距离不够时会自动减小半径。

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param end1: 第1个轴运动绝对坐标。  type:float

        :param end2: 第2个轴运动绝对坐标。  type:float

        :param end3: 第3个轴运动绝对坐标。  type:float

        :param next1: 第1个轴下一个直线运动绝对坐标。  type:float

        :param next2: 第2个轴下一个直线运动绝对坐标。  type:float

        :param next3: 第3个轴下一个直线运动绝对坐标。  type:float

        :param radius: 插入圆弧的半径，当过大的时候自动缩小。  type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_MoveSmoothSp(self.handle, ctypes.c_int(imaxaxises), Axislistarray,
                                               ctypes.c_float(end1), ctypes.c_float(end2), ctypes.c_float(end3),
                                               ctypes.c_float(next1), ctypes.c_float(next2), ctypes.c_float(next3),
                                               ctypes.c_float(radius))
        return ret

    def ZAux_Direct_MovePause(self, iaxis, imode):
        '''
        :Description:运动暂停		，插补运动暂停主轴。轴列表轴第一个轴。

        :param iaxis:轴号。  type:int

        :param imode:模式   type:int
                     0(缺省)暂停当前运动。
                     1 在当前运动完成后正准备执行下一条运动指令时暂停。
                     2 在当前运动完成后正准备执行下一条运动指令时,并且两条指令的MARK标识不一样时暂停。这个模式可以用于一个动作由多个指令来实现时,可以在一整个动作完成后暂停。

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MovePause(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_MoveResume(self, iaxis):
        '''
        :Description:取消运动暂停。

        :param iaxis:轴号。  type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveResume(self.handle, ctypes.c_int(iaxis))
        return ret

    def ZAux_Direct_MoveLimit(self, iaxis, limitspeed):
        '''
        :Description:在当前的运动末尾位置增加速度限制，用于强制拐角减速。

        :param iaxis:轴号。  type:int

        :param limitspeed:限制到的速度  type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveLimit(self.handle, ctypes.c_int(iaxis), ctypes.c_float(limitspeed))
        return ret

    def ZAux_Direct_MoveOp(self, iaxis, ioutnum, ivalue):
        '''
        :Description:在运动缓冲中加入输出指令。

        :param iaxis:轴号。  type:int

        :param ioutnum:输出口编号  type:int

        :param ivalue:输出口状态  type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveOp(self.handle, ctypes.c_int(iaxis), ctypes.c_int(ioutnum), ctypes.c_int(ivalue))
        return ret

    def ZAux_Direct_MoveOpMulti(self, iaxis, ioutnumfirst, ioutnumend, ivalue):
        '''
        :Description:在运动缓冲中加入连续输出口输出指令。

        :param iaxis:轴号。  type:int

        :param ioutnumfirst:输出口起始编号  type:int

        :param ioutnumend:输出口结束编号  type:int

        :param ivalue:对应输出口状态二进制组合值  type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveOpMulti(self.handle, ctypes.c_int(iaxis), ctypes.c_int(ioutnumfirst),
                                              ctypes.c_int(ioutnumend), ctypes.c_int(ivalue))
        return ret

    def ZAux_Direct_MoveOp2(self, iaxis, ioutnum, ivalue, iofftimems):
        '''
        :Description:在运动缓冲中加入输出指令 ,指定时间后输出状态翻转。

        :param iaxis:轴号。  type:int

        :param ioutnum:输出口起始编号  type:int

        :param ivalue:输出口状态  type:int

        :param iofftimems:状态反转时间  type:int

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_MoveOp2(self.handle, ctypes.c_int(iaxis), ctypes.c_int(ioutnum), ctypes.c_int(ivalue),
                                          ctypes.c_int(iofftimems))
        return ret

    def ZAux_Direct_MoveAout(self, iaxis, ioutnum, fvalue):
        '''
        :Description:在运动缓冲中加入输出指令 ,指定时间后输出状态翻转。

        :param iaxis:轴号。  type:int

        :param ioutnum:输出口起始编号  type:int

        :param ivalue:输出口状态  type:int

        :param iofftimems:状态反转时间  type:int

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_MoveAout(self.handle, ctypes.c_int(iaxis), ctypes.c_int(ioutnum),
                                           ctypes.c_int(fvalue))
        return ret

    def ZAux_Direct_MoveDelay(self, iaxis, itimems):
        '''
        :Description:在运动缓冲中加入延时指令。

        :param iaxis:轴号。  type:int

        :param itimems:延时时间 itimems 毫秒 。 type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MoveDelay(self.handle, ctypes.c_int(iaxis), ctypes.c_int(itimems))
        return ret

    def ZAux_Direct_MoveTurnabs(self, tablenum, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:旋转台直线插补运动。  20130901 以后的控制器版本支持。

        :param iaxis:轴号。  type:int

        :param tablenum:存储旋转台参数的table编号。   type:int

        :param imaxaxises:参与运动总轴数。  type:int

        :param piAxislist:轴号列表。  type:int

        :param pfDisancelist:距离列表。 type:float

        :Return:错误码。 type: int32

        '''
        piAxislistarry = (ctypes.c_float * len(piAxislist))(*piAxislist)
        pfDisancelistarry = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_MoveTurnabs(self.handle, ctypes.c_int(tablenum), ctypes.c_int(imaxaxises),
                                              piAxislistarry, pfDisancelistarry)
        return ret

    def ZAux_Direct_McircTurnabs(self, tablenum, refpos1, refpos2, mode, end1, end2, imaxaxises, piAxislist,
                                 pfDisancelist):
        '''
        :Description:旋转台圆弧+螺旋插补运动。  20130901 以后的控制器版本支持。

        :param iaxis:轴号。  type:int

        :param tablenum:存储旋转台参数的table编号。   type:int

        :param refpos1:第一个轴参考点，绝对位置。  type:float

        :param refpos2:第一个轴参考点，绝对位置。  type:float

        :param mode: 1-参考点是当前点前面,2-参考点是结束点后面,3-参考点在中间，采用三点定圆的方式。  type:int

        :param end1:第一个轴结束点，绝对位置。  type:float

        :param end2:第二个轴结束点，绝对位置。 type:float

        :param imaxaxises:参与运动轴数量。 type:int

        :param piAxislist:轴列表。 type:int

        :param pfDisancelist:螺旋轴距离列表。 type:float

        :Return:错误码。 type: int32

        '''
        piAxislistarry = (ctypes.c_float * len(piAxislist))(*piAxislist)
        pfDisancelistarry = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_McircTurnabs(self.handle, tablenum, refpos1, refpos2, mode, end1, end2, imaxaxises,
                                               piAxislistarry, pfDisancelistarry)
        return ret

    def ZAux_Direct_Cam(self, iaxis, istartpoint, iendpoint, ftablemulti, fDistance):
        '''
        :Description:电子凸轮 同步运动。

        :param iaxis:轴号。  type:int

        :param istartpoint:起始点TABLE编号。   type:int

        :param iendpoint:结束点TABLE编号。  type:int

        :param ftablemulti:位置比例，一般设为脉冲当量值。  type:float

        :param fDistance:参考运动的距离，用来计算总运动时间。 type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Cam(self.handle, ctypes.c_int(iaxis), ctypes.c_int(istartpoint),
                                      ctypes.c_int(iendpoint), ctypes.c_float(ftablemulti), ctypes.c_float(fDistance))
        return ret

    def ZAux_Direct_Cambox(self, iaxis, istartpoint, iendpoint, ftablemulti, fDistance, ilinkaxis, ioption,
                           flinkstartpos):
        '''
        :Description:电子凸轮 同步运动。

        :param iaxis:轴号。  type:int

        :param istartpoint:起始点TABLE编号。   type:int

        :param iendpoint:结束点TABLE编号。  type:int

        :param ftablemulti:位置比例，一般设为脉冲当量值。  type:float

        :param fDistance:参考运动的距离，用来计算总运动时间。 type:float

        :param ilinkaxis:参考主轴。 type:int

        :param ioption:参考轴的连接方式。 type:int

        :param flinkstartpos:ioption条件中距离参数。 type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Cambox(self.handle, ctypes.c_int(iaxis), ctypes.c_int(istartpoint),
                                         ctypes.c_int(iendpoint), ctypes.c_float(ftablemulti),
                                         ctypes.c_float(fDistance), ctypes.c_int(ilinkaxis), ctypes.c_int(ioption),
                                         ctypes.c_float(flinkstartpos))
        return ret

    def ZAux_Direct_Movelink(self, iaxis, fDistance, fLinkDis, fLinkAcc, fLinkDec, iLinkaxis, ioption, flinkstartpos):
        '''
        :Description:电子凸轮 同步运动。

        :param iaxis:参与运动的轴号(跟随轴)。  type:int

        :param fDistance:同步过程跟随轴运动距离。   type:float

        :param fLinkDis:同步过程参考轴(主轴)运动绝对距离。  type:float

        :param fLinkAcc:跟随轴加速阶段，参考轴移动的绝对距离。  type:float

        :param fLinkDec:跟随轴减速阶段，参考轴移动的绝对距离。 type:float

        :param iLinkaxis:参考轴的轴号。 type:int

        :param ioption:连接模式选项。 type:int

        :param flinkstartpos:连接模式选项中运动距离。 type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Movelink(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fDistance),
                                           ctypes.c_float(fLinkDis), ctypes.c_float(fLinkAcc), ctypes.c_float(fLinkDec),
                                           ctypes.c_int(iLinkaxis), ctypes.c_float(ioption),
                                           ctypes.c_float(flinkstartpos))
        return ret

    def ZAux_Direct_Moveslink(self, iaxis, fDistance, fLinkDis, startsp, endsp, iLinkaxis, ioption, flinkstartpos):
        '''
        :Description:特殊凸轮 同步运动。

        :param iaxis:参与运动的轴号(跟随轴)。  type:int

        :param fDistance:同步过程跟随轴运动距离。   type:float

        :param fLinkDis:同步过程参考轴(主轴)运动绝对距离。  type:float

        :param fLinkAcc:启动时跟随轴和参考轴的速度比例,units/units单位,负数表示跟随轴负向运动。  type:float

        :param fLinkDec:结束时跟随轴和参考轴的速度比例,units/units单位, 负数表示跟随轴负向运动。 type:float

        :param iLinkaxis:参考轴的轴号。 type:int

        :param ioption:连接模式选项。 type:int

        :param flinkstartpos:连接模式选项中运动距离。 type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Moveslink(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fDistance),
                                            ctypes.c_float(fLinkDis), ctypes.c_float(startsp), ctypes.c_float(endsp),
                                            ctypes.c_int(iLinkaxis), ctypes.c_int(ioption),
                                            ctypes.c_float(flinkstartpos))
        return ret

    def ZAux_Direct_Connect(self, ratio, link_axis, move_axis):
        '''
        :Description:连接 同步运动指令 电子齿轮。

        :param ratio:比率，可正可负，注意是脉冲个数的比例。  type:flot

        :param link_axis:连接轴的轴号，手轮时为编码器轴。   type:int

        :param move_axis:随动轴号。  type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Connect(self.handle, ctypes.c_float(ratio), ctypes.c_int(link_axis),
                                          ctypes.c_int(move_axis))
        return ret

    def ZAux_Direct_Connpath(self, ratio, link_axis, move_axis):
        '''
        :Description:连接 同步运动指令 电子齿轮 将当前轴的目标位置与link_axis轴的插补矢量长度通过电子齿轮连接。

        :param ratio:比率，可正可负，注意是脉冲个数的比例。  type:int

        :param link_axis:连接轴的轴号，手轮时为编码器轴。   type:int

        :param move_axis:随动轴号。  type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Connpath(self.handle, ctypes.c_float(ratio), ctypes.c_int(link_axis),
                                           ctypes.c_int(move_axis))
        return ret

    def ZAux_Direct_Regist(self, iaxis, imode):
        '''
        :Description:连接 同步运动指令 电子齿轮 将当前轴的目标位置与link_axis轴的插补矢量长度通过电子齿轮连接。

        :param iaxis:轴号。  type:int

        :param imode:锁存模式。   type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Regist(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_EncoderRatio(self, iaxis, output_count, input_count):
        '''
        :Description:编码器输入齿轮比，缺省(1,1)。

        :param iaxis:轴号。  type:int

        :param output_count:分子,不要超过65535。   type:int

        :param input_count:分母,不要超过65535。   type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_EncoderRatio(self.handle, ctypes.c_int(iaxis), ctypes.c_int(output_count),
                                               ctypes.c_int(input_count))
        return ret

    def ZAux_Direct_StepRatio(self, iaxis, output_count, input_count):
        '''
        :Description:设置步进输出齿轮比，缺省(1,1)。

        :param iaxis:轴号。  type:int

        :param output_count:分子,1-65535。   type:int

        :param input_count: 分母,1-65535。   type:int

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_StepRatio(self.handle, ctypes.c_int(iaxis), ctypes.c_int(output_count),
                                            ctypes.c_int(input_count))
        return ret

    def ZAux_Direct_Rapidstop(self, imode):
        '''
        :Description:所有轴立即停止。

        :param imode: 停止模式。  type:int
                      0(缺省)取消当前运动
				      1	取消缓冲的运动
				      2	取消当前运动和缓冲运动。
				      3	立即中断脉冲发送。

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Rapidstop(self.handle, ctypes.c_int(imode))
        return ret

    def ZAux_Direct_CancelAxisList(self, imaxaxises, piAxislist, imode):
        '''
        :Description:多个轴运动停止。

        :param imaxaxises:轴数。   type:int

        :param piAxislist:轴列表。   type:int

        :param imode: 停止模式。  type:int
                      0(缺省)取消当前运动
				      1	取消缓冲的运动
				      2	取消当前运动和缓冲运动。
				      3	立即中断脉冲发送。

        :Return:错误码。 type: int32

        '''
        Axisarry = (ctypes.c_int * len(piAxislist))(*piAxislist)
        ret = zauxdll.ZAux_Direct_CancelAxisList(self.handle, ctypes.c_int(imaxaxises), Axisarry, ctypes.c_int(imode))
        return ret

    def ZAux_Direct_Connframe(self, Jogmaxaxises, JogAxislist, frame, tablenum, Virmaxaxises, VirAxislist):
        '''
        :Description:CONNFRAME机械手逆解指令	2系列以上控制器支持。
。
        :param Jogmaxaxises:关节轴数量。   type:int

        :param JogAxislist:关节轴列表。   type:int

        :param frame:机械手类型。   type:int

        :param tablenum:机械手参数TABLE起始编号。   type:int

        :param Virmaxaxises:关联虚拟轴个数。   type:int

        :param VirAxislist:虚拟轴列表。   type:int

        :Return:错误码。 type: int32

        '''
        JogAxislistarry = (ctypes.c_int * len(JogAxislist))(*JogAxislist)
        VirAxislistarry = (ctypes.c_int * len(VirAxislist))(*VirAxislist)
        ret = zauxdll.ZAux_Direct_Connframe(self.handle, ctypes.c_int(Jogmaxaxises), JogAxislistarry,
                                            ctypes.c_int(frame), ctypes.c_int(tablenum), ctypes.c_int(Virmaxaxises),
                                            VirAxislistarry)
        return ret

    def ZAux_Direct_Connreframe(self, Virmaxaxises, VirAxislist, frame, tablenum, Jogmaxaxises, JogAxislist):
        '''
        :Description:CONNREFRAME机械手正解指令	2系列以上控制器支持。

        :param Virmaxaxises:关联虚拟轴个数。   type:int

        :param VirAxislist:虚拟轴列表。   type:int

        :param frame:机械手类型。   type:int

        :param tablenum:机械手参数TABLE起始编号。   type:int

        :param Jogmaxaxises:关节轴数量。   type:int

        :param JogAxislist:关节轴列表。   type:int

        :Return:错误码。 type: int32

        '''
        JogAxislistarry = (ctypes.c_int * len(JogAxislist))(*JogAxislist)
        VirAxislistarry = (ctypes.c_int * len(VirAxislist))(*VirAxislist)
        ret = zauxdll.ZAux_Direct_Connreframe(self.handle, Virmaxaxises, VirAxislistarry, frame, tablenum, Jogmaxaxises,
                                              JogAxislistarry)
        return ret

    def ZAux_Direct_Single_Addax(self, iaxis, iaddaxis):
        '''
        :Description:轴叠加运动	iaddaxis运动叠加到iaxis轴 ,ADDAX指令叠加的是脉冲个数。

        :param iaxis:被叠加轴   type:int

        :param iaddaxis:叠加轴。   type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_Addax(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iaddaxis))
        return ret

    def ZAux_Direct_Single_Cancel(self, iaxis, imode):
        '''
        :Description:单轴运动停止

        :param iaxis:轴号。   type:int

        :param imode: 停止模式。  type:int
                      0(缺省)取消当前运动
				      1	取消缓冲的运动
				      2	取消当前运动和缓冲运动。
				      3	立即中断脉冲发送。

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_Cancel(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_Single_Vmove(self, iaxis, idir):
        '''
        :Description:单轴连续运动。

        :param iaxis:轴号。   type:int

        :param idir:方向 1正向 -1负向。   type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_Vmove(self.handle, ctypes.c_int(iaxis), ctypes.c_int(idir))
        return ret

    def ZAux_Direct_Single_Datum(self, iaxis, imode):
        '''
        :Description:控制器方式回零。

        :param iaxis:轴号。   type:int

        :param imode:模式。   type:int

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_Datum(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_GetHomeStatus(self, iaxis):
        '''
        :Description:回零完成状态。

        :param iaxis:轴号。   type:int

        :Return,:错误码,回零完成标志 0-回零异常 1回零成功。 type: int32,uint32

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetHomeStatus(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_Single_Move(self, iaxis, fdistance):
        '''
        :Description:单轴相对运动。

        :param iaxis:轴号。   type:int

        :param fdistance:距离。   type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_Move(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fdistance))
        return ret

    def ZAux_Direct_Single_MoveAbs(self, iaxis, fdistance):
        '''
        :Description:单轴绝对运动。

        :param iaxis:轴号。   type:int

        :param fdistance:距离。   type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_Single_MoveAbs(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fdistance))
        return ret

    def ZAux_Direct_SetVrf(self, vrstartnum, numes, pfValue):
        '''
        :Description:写VR。

        :param vrstartnum:VR起始编号。   type:int

        :param numes:写入的数量。   type:int

        :param pfValue:写入的数据列表。   type:float

        :Return:错误码。 type: int32

        '''
        pfValuearry = (ctypes.c_float * len(pfValue))(*pfValue)
        ret = zauxdll.ZAux_Direct_SetVrf(self.handle, ctypes.c_int(vrstartnum), ctypes.c_int(numes), pfValuearry)
        return ret

    def ZAux_Direct_GetVrf(self, vrstartnum, numes):
        '''
        :Description:VR读取, 可以一次读取多个。

        :param vrstartnum:VR起始编号。   type:int

        :param numes:写入的数量。   type:int

        :Return:错误码,返回的读取值，多个时必须分配空间。 type: int32,float

        '''
        value = (ctypes.c_float * numes)()
        ret = zauxdll.ZAux_Direct_GetVrf(self.handle, ctypes.c_int(vrstartnum), ctypes.c_int(numes),
                                         ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetVrInt(self, vrstartnum, numes):
        '''
        :Description:VRINT读取, 必须150401以上版本才支持VRINT的DIRECTCOMMAND读取。

        :param vrstartnum:VR起始编号。   type:int

        :param numes:读取的数量。   type:int

        :Return:错误码,返回的读取值，多个时必须分配空间。 type: int32,float

        '''
        value = (ctypes.c_int * numes)()
        ret = zauxdll.ZAux_Direct_GetVrf(self.handle, ctypes.c_int(vrstartnum), ctypes.c_int(numes),
                                         ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetTable(self, tabstart, numes, pfValue):
        '''
        :Description:写table 。

        :param tabstart:写入的TABLE起始编号。   type:int

        :param numes:写入的数量。   type:int

        :param pfValue:写入的数据值。   type:float

        :Return:错误码。       type: int32

        '''
        value = (ctypes.c_float * len(pfValue))(*pfValue)
        ret = zauxdll.ZAux_Direct_SetTable(self.handle, ctypes.c_int(tabstart), ctypes.c_int(numes), value)
        return ret

    def ZAux_Direct_GetTable(self, tabstart, numes):
        '''
        :Description:table读取, 可以一次读取多个。

        :param tabstart:读取TABLE起始地址。   type:int

        :param numes:读取的数量。   type:int

        :Return:错误码,返回的读取值，多个时必须分配空间。 type: int32,float

        '''
        value = (ctypes.c_float * numes)()
        ret = zauxdll.ZAux_Direct_GetTable(self.handle, ctypes.c_int(tabstart), ctypes.c_int(numes),
                                           ctypes.byref(value))
        return ret, value

    def ZAux_TransStringtoFloat(self, pstringin, inumes):
        '''
        :Description:字符串转为float。

        :param pstringin:数据的字符串。   type:sting

        :param inumes: 转换数据个数。   type:int

        :Return:错误码,转换的数据。 type: int32,float

        '''
        _str = pstringin.encode('utf-8')
        value = (ctypes.c_float * inumes)()
        ret = zauxdll.ZAux_TransStringtoFloat(_str, inumes, ctypes.byref(value))
        return ret, value

    def ZAux_TransStringtoInt(self, pstringin, inumes):
        '''
        :Description:字符串转为int。

        :param pstringin:数据的字符串。   type:sting

        :param inumes: 转换数据个数。   type:int

        :Return:错误码,转换的数据。 type: int32,int

        '''
        _str = pstringin.encode('utf-8')
        value = (ctypes.c_int * inumes)()
        ret = zauxdll.ZAux_TransStringtoInt(_str, inumes, ctypes.byref(value))
        return ret, value

    def ZAux_WriteUFile(self, sFilename, pVarlist, inum):
        '''
        :Description:把float格式的变量列表存储到文件,与控制器的U盘文件格式一致。

        :param sFilename:文件绝对路径。   type:sting

        :param pVarlist:写入的数据列表。   type:float

        :Return:错误码。 type: int32

        '''
        _str = sFilename.encode('utf-8')
        value = (ctypes.c_float * len(pVarlist))(*pVarlist)
        ret = zauxdll.ZAux_WriteUFile(_str, value, len(pVarlist), ctypes.c_int(inum))
        return ret

    def ZAux_ReadUFile(self, sFilename, inum):
        '''
        :Description:读取float格式的变量列表,与控制器的U盘文件格式一致。

        :param sFilename:文件绝对路径。   type:sting

        :Return:错误码,读取的数据列表。 type: int32,int

        '''
        _str = sFilename.encode('utf-8')
        value = (ctypes.c_float * (inum))()
        ret = zauxdll.ZAux_ReadUFile(_str, ctypes.byref(value))
        return ret, value

    def ZAux_Modbus_Set0x(self, start, inum, pdata):
        '''
        :Description:modbus寄存器操作 modbus_bit。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :param pdata:设置的位状态(列表类型)。   type:uint8

        :Return:错误码。 type: uint16

        '''
        Axislistarray = (ctypes.c_int * len(pdata))(*pdata)
        ret = zauxdll.ZAux_Modbus_Set0x(self.handle, ctypes.c_uint(start), ctypes.c_uint(inum), Axislistarray)
        return ret

    def ZAux_Modbus_Get0x(self, start, inum):
        '''
        :Description:modbus寄存器操作 modbus_bit。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :Return:错误码,返回的位状态  按位存储。 type: int32,uint8

        '''
        value = (ctypes.c_uint8 * inum)()
        ret = zauxdll.ZAux_Modbus_Get0x(self.handle, ctypes.c_uint(start), ctypes.c_uint(inum), ctypes.byref(value))
        return ret, value

    def ZAux_Modbus_Set4x(self, start, inum, pdata):
        '''
        :Description:modbus寄存器操作 MODBUS_REG。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

         :param inum:设置值。   type:uint16

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int * len(pdata))(*pdata)
        ret = zauxdll.ZAux_Modbus_Set4x(self.handle, ctypes.c_uint(start), ctypes.c_uint(inum), Axislistarray)
        return ret

    def ZAux_Modbus_Get4x(self, start, inum):
        '''
        :Description:modbus寄存器操作  MODBUS_REG。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :Return:错误码,读取的REG寄存器值。 type: int32,uint16

        '''
        value = (ctypes.c_int16 * inum)()
        ret = zauxdll.ZAux_Modbus_Get4x(self.handle, ctypes.c_uint(start), ctypes.c_uint(inum), ctypes.byref(value))
        return ret, value

    def ZAux_Modbus_Get4x_Float(self, start, inum):
        '''
        :Description:Modbus 寄存器操作 MODBUS_IEEE 读取。		MODBUS_IEEE。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :Return:错误码,读取的REG寄存器值。 type: int32,float

        '''
        value = (ctypes.c_float * inum)()
        ret = zauxdll.ZAux_Modbus_Get4x_Float(self.handle, ctypes.c_uint(start), ctypes.c_uint(inum),
                                              ctypes.byref(value))
        return ret, value

    def ZAux_Modbus_Set4x_Float(self, start, inum, pdata):
        '''
        :Description:        :modbus寄存器操作。		MODBUS_IEEE

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

         :param pdata:数据列表。   type:float

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_float * len(pdata))(*pdata)
        ret = zauxdll.ZAux_Modbus_Set4x_Float(self.handle, ctypes.c_uint(start), ctypes.c_int(inum), Axislistarray)
        return ret

    def ZAux_Modbus_Get4x_Long(self, start, inum):
        '''
        :Description:        :modbus寄存器操作		MODBUS_LONG。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :Return:错误码,读取的REG寄存器值。 type: int32,int32

        '''

        value = (ctypes.c_int32 * inum)()
        ret = zauxdll.ZAux_Modbus_Get4x_Long(self.handle, ctypes.c_uint(start), ctypes.c_int(inum), ctypes.byref(value))
        return ret, value

    def ZAux_Modbus_Set4x_Long(self, start, inum, pidata):
        '''
        :Description:        :modbus寄存器操作		MODBUS_LONG。

        :param start:起始编号。   type:uint16

        :param inum:数量。   type:uint16

        :param inum:设置值。   type:int32

        :Return:错误码。 type: int32

        '''
        Axislistarray = (ctypes.c_int32 * len(pidata))(*pidata)
        ret = zauxdll.ZAux_Modbus_Set4x_Long(self.handle, ctypes.c_uint(start), ctypes.c_int(inum), Axislistarray)
        return ret

    def ZAux_Modbus_Set4x_String(self, start, inum, pdata):
        '''
        :Description:设置modbus_string。

        :param start:modbus起始地址。   type:uint16

        :param inum:长度。   type:uint16

        :param pdata:写入的字符串。   type:sting

        :Return:错误码。 type: int32

        '''
        _str = pdata.encode('utf-8')
        ret = zauxdll.ZAux_Modbus_Set4x_String(self.handle, ctypes.c_uint(start), ctypes.c_int(inum), _str)
        return ret

    def ZAux_Modbus_Get4x_String(self, start, inum):
        '''
        :Description:读取modbus_string。

        :param start:modbus起始地址。   type:uint16

        :param inum:长度。   type:uint16

        :Return:错误码,读取返回的字符串。 type: int32,sting

        '''
        value = (ctypes.c_char * inum)()
        ret = zauxdll.ZAux_Modbus_Get4x_String(self.handle, ctypes.c_uint(start), ctypes.c_int(inum),
                                               ctypes.byref(value))
        return ret, value

    def ZAux_FlashWritef(self, uiflashid, uinumes, pfvlue):
        '''
        :Description:写用户flash块, float数据。

        :param uiflashid:modbus起始地址。   type:uint16

        :param uinumes:变量个数。   type:int32

        :param pfvlue:数据列表。   type:float

        :Return:错误码。 type: int32,int32

        '''
        Axislistarray = (ctypes.c_float * len(pfvlue))(*pfvlue)
        ret = zauxdll.ZAux_FlashWritef(self.handle, ctypes.c_uint16(uiflashid), ctypes.c_int32(uinumes), Axislistarray)
        return ret

    def ZAux_FlashReadf(self, uiflashid, uinumes):
        '''
        :Description:读取用户flash块, float数据。

        :param uiflashid:flash块号。   type:uint16

        :param uinumes:缓冲变量个数。   type:int32

        :Return:错误码,读取到的变量个数。 type: int32,int32

        '''
        value = (ctypes.c_float * uinumes)()
        ret = zauxdll.ZAux_FlashReadf(self.handle, ctypes.c_uint16(uiflashid), ctypes.c_int32(uinumes),
                                      ctypes.byref(value))
        return ret, value

    def ZAux_Trigger(self):
        '''
        :Description:示波器触发函数 150723以后固件版本支持。

        '''
        ret = zauxdll.ZAux_Trigger(self.handle)
        return ret

    def ZAux_Direct_MovePara(self, base_axis, paraname, iaxis, fvalue):
        '''
        :Description:运动中修改参数. 20170503以上固件支持。

        :param base_axis:运动轴轴号。   type:uint32

        :param paraname:参数名称字符串。   type:sting

        :param iaxis:修改参数的轴号。   type: uint32

        :param fvalue:参数设定值。   type:float

        :Return:错误码。 type: int32

        '''
        _str = paraname.encode('utf-8')
        ret = zauxdll.ZAux_Direct_MovePara(self.handle, ctypes.c_uint32(base_axis), _str, ctypes.c_int32(iaxis),
                                           ctypes.c_float(fvalue))
        return ret

    def ZAux_Direct_MovePwm(self, base_axis, pwm_num, pwm_duty, pwm_freq):
        '''
        :Description:运动中修改PWM 20170503以上固件支持。

        :param base_axis:运动轴轴号。   type:uint32

        :param pwm_num:PWM 口编号。   type:uint32

        :param pwm_duty:设定的占空比。   type: float

        :param pwm_freq:设定的频率。   type:float

        :Return:错误码。 type: int32

        '''

        ret = zauxdll.ZAux_Direct_MovePwm(self.handle, ctypes.c_uint32(base_axis), ctypes.c_uint32(pwm_num),
                                          ctypes.c_float(pwm_duty), ctypes.c_float(pwm_freq))
        return ret

    def ZAux_Direct_MoveASynmove(self, base_axis, iaxis, fdist, ifsp):
        '''
        :Description:运动中触发其他轴的运动. 20170503以上固件支持。

        :param base_axis:运动轴轴号。   type:uint32

        :param iaxis:触发的轴号。   type:uint32

        :param fdist:触发运动的距离。   type: float

        :param ifsp:触发运动是否为 SP 运动。   type:uint32

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_MoveASynmove(self.handle, ctypes.c_uint32(base_axis), ctypes.c_uint32(iaxis),
                                               ctypes.c_float(fdist), ctypes.c_int32(ifsp))
        return ret

    def ZAux_Direct_MoveTable(self, base_axis, table_num, fvalue):
        '''
        :Description:运动中修改TABLE。

        :param base_axis:插补主轴编号。   type:uint32

        :param table_num:TABLE编号。   type:uint32

        :param fvalue:修改值。   type: float

        :Return:错误码。 type: int32

        '''
        ret = zauxdll.ZAux_Direct_MoveTable(self.handle, ctypes.c_uint32(base_axis), ctypes.c_uint32(table_num),
                                            ctypes.c_float(fvalue))
        return ret

    def ZAux_Direct_MoveWait(self, base_axis, paraname, inum, Cmp_mode, fvalue):
        '''
        :Description:BASE轴运动缓冲加入一个可变的延时  固件150802以上版本, 或XPLC160405以上版本支持。

        :param base_axis:插补主轴编号。    type: uint32

        :param paraname:参数名字符串 DPOS MPOS IN AIN VPSPEED MSPEED MODBUS_REG MODBUS_IEEE MODBUS_BIT NVRAM VECT_BUFFED  REMAIN 。 type: string

        :param inum:参数编号或轴号。       type: int

        :param Cmp_mode:比较条件 1 >=   0=  -1<=  对IN等BIT类型参数无效。      type: int

        :param fvalue:比较值。       type: float

        :Return:错误码。             type: int32

        '''
        _str = paraname.encode('utf-8')
        ret = zauxdll.ZAux_Direct_MoveWait(self.handle, ctypes.c_uint32(base_axis), _str, ctypes.c_int(inum),
                                           ctypes.c_int(Cmp_mode), ctypes.c_float(fvalue))
        return ret

    def ZAux_Direct_MoveTask(self, base_axis, tasknum, labelname):
        '''
        :Description:BASE轴运动缓冲加入一个TASK任务 当任务已经启动时，会报错，但不影响程序执行。

        :param base_axis:插补主轴编号。    type: uint32

        :param tasknum:任务编号 。 type: uint32

        :param labelname:BAS中全局函数名或者标号。       type: sting

        :Return:错误码。             type: int32

        '''
        _str = labelname.encode('utf-8')
        ret = zauxdll.ZAux_Direct_MoveTask(self.handle, ctypes.c_uint32(base_axis), ctypes.c_uint32(tasknum), _str)
        return ret

    def ZAux_Direct_Pswitch(self, num, enable, axisnum, outnum, outstate, setpos, resetpos):
        '''
        :Description:位置比较PSWITCH。

        :param num:比较器编号  0-15。    type: int

        :param enable:比较器使能 0/1。    type: int

        :param axisnum:比较的轴号。        type: int

        :param outnum:输出口编号。          type: int

        :param outstate:输出状态 0/1 。       type: int

        :param setpos:比较起始位置。         type: float

        :param resetpos:输出复位位置。       type: float

        :Return:错误码。             type: int32

        '''

        ret = zauxdll.ZAux_Direct_Pswitch(self.handle, ctypes.c_int(num), ctypes.c_int(enable), ctypes.c_int(axisnum),
                                          ctypes.c_int(outnum), ctypes.c_int(outstate), ctypes.c_float(setpos),
                                          ctypes.c_float(resetpos))
        return ret

    def ZAux_Direct_HwPswitch(self, Axisnum, Mode, Direction, Reserve, Tablestart, Tableend):
        '''
        :Description:硬件位置比较输出 4系列产品脉冲轴与编码器轴支持硬件比较输出。

        :param Axisnum:比较输出的轴号。    type: int

        :param Mode:比较器操作 1-启动比较器 2-停止并删除未完成的点。    type: int

        :param Direction:比较方向 0-负向 1-正向。        type: int

        :param Reserve:预留。          type: int

        :param Tablestart:TABLE 起始地址 。       type: int

        :param Tableend:TABLE 结束地址。         type: int

        :Return:错误码。             type: int32

        '''

        ret = zauxdll.ZAux_Direct_HwPswitch(self.handle, ctypes.c_int(Axisnum), ctypes.c_int(Mode),
                                            ctypes.c_int(Direction), ctypes.c_int(Reserve), ctypes.c_int(Tablestart),
                                            ctypes.c_int(Tableend))
        return ret

    def ZAux_Direct_GetHwPswitchBuff(self, axisnum):
        '''
        :Description:硬件位置比较输出剩余缓冲获取 4系列产品脉冲轴与编码器轴支持硬件比较输出。

        :param axisnum:比较输出的轴号。    type: int

        :Return:错误码,位置比较输出剩余缓冲数。             type: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetHwPswitchBuff(self.handle, ctypes.c_int(axisnum), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_HwTimer(self, mode, cyclonetime, optime, reptimes, opstate, opnum):
        '''
        :Description:硬件定时器用于硬件比较输出后一段时间后还原电平 4系列产品支持。

        :param mode:模式。    type: int

        :param cyclonetime:周期时间 us单位。    type: int

        :param optime:有效时间 us单位。        type: int

        :param reptimes:重复次数。          type: int

        :param opstate:输出缺省状态   输出口变为非此状态后开始计时。       type: int

        :param opnum:输出口编号 必须能硬件比较输出的口。         type: int

        :Return:错误码。             type: int32

        '''

        ret = zauxdll.ZAux_Direct_HwTimer(self.handle, ctypes.c_int(mode), ctypes.c_int(cyclonetime),
                                          ctypes.c_int(optime), ctypes.c_int(reptimes), ctypes.c_int(opstate),
                                          ctypes.c_int(opnum))
        return ret

    def ZAux_Direct_GetAxisStopReason(self, iaxis):
        '''
        :Description:读取轴停止原因。

        :param iaxis:轴号。    type: int

        :Return:错误码,返回状态值，对应的位表示不同状态。             type: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetAxisStopReason(self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetAllAxisPara(self, sParam, imaxaxis):
        '''
        :Description:浮点型读全部轴参数状态。

        :param sParam:Baisc 语法参数的字符串名称。    type: sting

        :param imaxaxis:轴数量。    type: int

        :Return:错误码,返回状态值。             type: int32,float

        '''
        _str = sParam.encode('utf-8')
        value = (ctypes.c_float * imaxaxis)()
        ret = zauxdll.ZAux_Direct_GetAllAxisPara(self.handle, _str, ctypes.c_int(imaxaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetAllAxisInfo(self, max_axis, idle_status, target_pos_status, feedback_pos_status, axis_status):
        max_axis = ctypes.c_int(max_axis)
        '''
        :Description:一次性读取多个轴轴参数。
    
        :param max_axis:轴数量。    type: int

        :param idle_status:轴运行状态。    type: int

        :param target_pos_status:轴命令坐标。
        
        :param feedback_pos_status:轴反馈坐标。
        
        :param axis_status:轴状态。

        :Return:错误码,返回状态值。             type: int32,float   

        '''
        # 所需参数皆为数组指针,所以需要在函数外部定义c的函数指针变量
        # idle_status = ctypes.pointer(ctypes.c_int(idle_status))
        # target_pos_status = ctypes.pointer(ctypes.c_float(target_pos_status))
        # feedback_pos_status = ctypes.pointer(ctypes.c_float(feedback_pos_status))
        # axis_status = ctypes.pointer(ctypes.c_int(axis_status))
        ret = zauxdll.ZAux_Direct_GetAllAxisInfo(self.handle, max_axis, idle_status, target_pos_status,
                                                 feedback_pos_status, axis_status)

    def ZAux_Direct_SetUserArray(self, arrayname, arraystart, numes, pfValue):
        '''
        :Description:设置BASIC自定义全局数组。

        :param arrayname:数组名称。       type: sting

        :param arraystart:数组起始元素 us单位。    type: int

        :param numes:元素数量。        type: int

        :param pfValue:设置值。          type: float

        :Return:错误码。             type: int32

        '''
        _str = arrayname.encode('utf-8')
        ARRYY = (ctypes.c_float * len(pfValue))(*pfValue)
        ret = zauxdll.ZAux_Direct_SetUserArray(self.handle, _str, ctypes.c_int(arraystart), ctypes.c_int(numes), ARRYY)
        return ret

    def ZAux_Direct_GetUserArray(self, arrayname, arraystart, numes):
        '''
        :Description:读取设置BASIC自定义全局数组 , 可以一次读取多个。

        :param arrayname:数组名称。       type: sting

        :param arraystart:数组起始元素 us单位。    type: int

        :param numes:元素数量。        type: int

        :Return:错误码,读取数组元素的值多个时必须分配空间。  type: int32,float

        '''
        _str = arrayname.encode('utf-8')
        value = (ctypes.c_float * numes)()
        ret = zauxdll.ZAux_Direct_GetUserArray(self.handle, _str, ctypes.c_int(arraystart), ctypes.c_int(numes),
                                               ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetUserVar(self, varname, pfValue):
        '''
        :Description:设置自定义变量。

        :param varname:变量名称字符串。       type: sting

        :param pfValue:变量值。       type: float

        :Return:错误码。             type: int32

        '''
        _str = varname.encode('utf-8')
        ret = zauxdll.ZAux_Direct_SetUserVar(self.handle, _str, ctypes.c_float(pfValue))
        return ret

    def ZAux_Direct_GetUserVar(self, varname):
        '''
        :Description:读取自定义全局变量。

        :param varname:变量名称字符串。       type: sting

        :Return:错误码,变量值。             type: int32,float

        '''
        _str = varname.encode('utf-8')
        value = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetUserVar(self.handle, _str, ctypes.byref(value))
        return ret, value

    def ZAux_OpenPci(self, cardnum):
        '''
        :Description:与控制器建立链接。

        :param cardnum:PCI卡号。       type: uint32

        :Return:错误码。             type: int32

        '''
        ret = zauxdll.ZAux_OpenPci(ctypes.c_int(cardnum), ctypes.pointer(self.handle))
        return ret

    def ZAux_BusCmd_GetNodeNum(self, slot):
        '''
        :Description:读取卡槽上节点数量。

        :param slot:槽位号缺省0。       type: int

        :Return:错误码， 返回扫描成功节点数量。type: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetNodeNum(self.handle, ctypes.c_int(slot), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_GetNodeInfo(self, slot, node, sel):
        '''
        :Description:读取节点上的信息。

        :param slot:槽位号缺省0。       type: int

        :param node:节点编号0。         type: int

        :param sel:信息编号	0-厂商编号1-设备编号 2-设备版本 3-别名 10-IN个数 11-OUT个数 。 type: int

        :Return:错误码,返回信息。type: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetNodeInfo(self.handle, ctypes.c_int(slot), ctypes.c_int(node), ctypes.c_int(sel),
                                              ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_GetNodeStatus(self, slot, node):
        '''
        :Description:读取节点总线状态。

        :param slot:槽位号缺省0。       type: int

        :param node:节点编号0。         type: int

        :Return:错误码, 按位处理 bit0-节点是否存在  bit1-通讯状态   bit2-节点状态。type: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetNodeStatus(self.handle, ctypes.c_int(slot), ctypes.c_int(node),
                                                ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_SDORead(self, slot, node, index, subindex, aype):
        '''
        :Description:读取节点SDO参数信息。

        :param slot:槽位号缺省0。       type: uint32

        :param node:节点编号0。       type: uint32

        :param index:对象字典编号(注意函数为10进制数据)0。       type: uint32

        :param subindex:子编号	(注意函数为10进制数据)。         type: uint32

        :param aype:数据类型  1-bool 2-int8 3-int16 4-int32 5-uint8 6-uint16 7-uint32。         type: uint32

        :Return:错误码, 读取的数据值: int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_SDORead(self.handle, ctypes.c_int(slot), ctypes.c_int(node), ctypes.c_int(index),
                                          ctypes.c_int(subindex), ctypes.c_int(aype), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_SDOWrite(self, slot, node, index, subindex, aype, Vvalue):
        '''
        :Description:写节点SDO参数信息。

        :param slot:槽位号缺省0。       type: uint32

        :param node:节点编号0。       type: uint32

        :param index:对象字典编号(注意函数为10进制数据)0。       type:uint32

        :param subindex:子编号	(注意函数为10进制数据)。         type: uint32

        :param aype:数据类型  1-bool 2-int8 3-int16 4-int32 5-uint8 6-uint16 7-uint32。         type: int

        :param Vvalue:设定的数据值。         type: uint32

        :Return:错误码: int32

        '''
        ret = zauxdll.ZAux_BusCmd_SDOWrite(self.handle, ctypes.c_int(slot), ctypes.c_int(node), ctypes.c_int(index),
                                           ctypes.c_int(subindex), ctypes.c_int(aype), ctypes.c_int(Vvalue))
        return ret

    def ZAux_BusCmd_SDOReadAxis(self, iaxis, index, subindex, aype):
        '''
        :Description:通过轴号进行 SDO 读取。

        :param iaxis:轴号。       type: uint32

        :param index:对象字典编号。       type: uint32

        :param subindex:对象字典子编号。       type: uint32

        :param aype:数据类型  1-bool 2-int8 3-int16 4-int32 5-uint8 6-uint16 7-uint32。         type: uint32

        :Return:错误码, 读取的数据值。 int32,int32

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_SDOReadAxis(self.handle, ctypes.c_int(iaxis), ctypes.c_int(index),
                                              ctypes.c_int(subindex), ctypes.c_int(aype), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_SDOWriteAxis(self, iaxis, index, subindex, aype, Vvalue):
        '''
        :Description:通过轴号进行 SDO 写入。

        :param iaxis:轴号。       type: uint32

        :param index:对象字典编号。       type: uint32

        :param subindex:对象字典子编号。       type: uint32

        :param aype:数据类型  1-bool 2-int8 3-int16 4-int32 5-uint8 6-uint16 7-uint32。         type: uint32

        :param Vvalue:对象字典子编号。       type: uint32

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_SDOWriteAxis(self.handle, ctypes.c_int(iaxis), ctypes.c_int(index),
                                               ctypes.c_int(subindex), ctypes.c_int(aype), ctypes.c_int(Vvalue))
        return ret

    def ZAux_BusCmd_RtexRead(self, iaxis, ipara):
        '''
        :Description:Rtex读取参数信息。

        :param iaxis:轴号。       type: uint32

        :param ipara:参数分类*256 + 参数编号  Pr7.11-ipara = 7*256+11。 type: uint32

        :Return:错误码,读取的数据值。    type:int32,float

        '''
        value = (ctypes.c_float)()
        ret = zauxdll.ZAux_BusCmd_RtexRead(self.handle, ctypes.c_uint(iaxis), ctypes.c_int32(ipara),
                                           ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_RtexWrite(self, iaxis, ipara, vvalue):
        '''
        :Description:Rtex写参数信息。

        :param iaxis:轴号。       type: uint32

        :param ipara:参数分类*256 + 参数编号  Pr7.11-ipara = 7*256+11。 type: uint32

        :param vvalue:设定的数据值。       type: float

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_RtexWrite(self.handle, ctypes.c_uint(iaxis), ctypes.c_uint(ipara),
                                            ctypes.c_float(vvalue))
        return ret

    def ZAux_BusCmd_SetDatumOffpos(self, iaxis, fValue):
        '''
        :Description:设置回零偏移距离。

        :param iaxis:轴号。       type: uint32

        :param fValue:偏移距离。 type: float

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_SetDatumOffpos(self.handle, ctypes.c_uint(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_BusCmd_GetDatumOffpos(self, iaxis):
        '''
        :Description:读取回零偏移距离。

        :param iaxis:轴号。       type: uint32

        :Return:错误码,偏移距离。    type:int32,float

        '''
        value = (ctypes.c_float)()
        ret = zauxdll.ZAux_BusCmd_GetDatumOffpos(self.handle, ctypes.c_uint(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_Datum(self, iaxis, homemode):
        '''
        :Description:总线驱动器回零。

        :param iaxis:轴号。       type: uint32

        :param homemode:回零模式，查看驱动器手册。       type: uint32

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_Datum(self.handle, ctypes.c_uint(iaxis), ctypes.c_uint(homemode))
        return ret

    def ZAux_BusCmd_GetHomeStatus(self, iaxis):
        '''
        :Description:驱动器回零完成状态。

        :param iaxis:轴号。       type: uint32

        :Return:错误码,零完成标志 0-回零异常 1回零成功。    type:int32,uint32

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetHomeStatus(self.handle, ctypes.c_uint(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_DriveClear(self, iaxis, mode):
        '''
        :Description:设置清除总线伺服报警。

        :param iaxis:轴号。       type: uint32

        :param mode:模式 0-清除当前告警  1-清除历史告警  2-清除外部输入告警。       type: uint32

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_DriveClear(self.handle, ctypes.c_uint(iaxis), ctypes.c_uint(mode))
        return ret

    def ZAux_BusCmd_GetDriveTorque(self, iaxis):
        '''
        :Description:读取当前总线驱动当前力矩	需要设置对应的DRIVE_PROFILE类型。

        :param iaxis:轴号。       type: int

        :Return:错误码,当前转矩。    type:int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetDriveTorque(self.handle, ctypes.c_uint(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_SetMaxDriveTorque(self, iaxis, piValue):
        '''
        :Description:设置当前总线驱动最大转矩  需要设置对应的DRIVE_PROFILE类型。

        :param iaxis:轴号。       type: int

         :param piValue:最大转矩限制。       type: int

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_SetMaxDriveTorque(self.handle, ctypes.c_uint(iaxis), ctypes.c_uint(piValue))
        return ret

    def ZAux_BusCmd_GetMaxDriveTorque(self, iaxis):
        '''
        :Description:读取当前总线驱动最大转矩  需要设置对应的DRIVE_PROFILE类型。

        :param iaxis:轴号。       type: int

        :Return:错误码,返回的最大转矩。    type:int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetMaxDriveTorque(self.handle, ctypes.c_uint(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetDAC(self, iaxis, fValue):
        '''
        :Description:设置模拟量输出 力矩、速度模式下可以  总线驱动需要设置对应DRIVE_PROFILE类型 与ATYPE

        :param iaxis:轴号。       type: int

         :param piValue:模拟量 输出值。       type: float

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_Direct_SetDAC(self.handle, ctypes.c_uint(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_GetDAC(self, iaxis):
        '''
        :Description:读取模拟量输出 力矩、速度模式下可以  总线驱动需要设置对应DRIVE_PROFILE类型 与ATYPE。

        :param iaxis:轴号。       type: int

        :Return:错误码,模拟量返回值。    type:int32,float

        '''
        value = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetDAC(self.handle, ctypes.c_uint(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_BusCmd_InitBus(self):
        '''
        :Description:总线初始化  (针对Zmotion tools 工具软件配置过总线参数控制器使用有效）。

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_BusCmd_InitBus(self.handle)
        return ret

    def ZAux_BusCmd_GetInitStatus(self):
        '''
        :Description:获取总线初始化完成状态(针对Zmotion tools 工具软件配置过总线参数控制器使用有效）。

        :Return:错误码,0-初始化失败 1成功。    type:int32,int

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_BusCmd_GetInitStatus(self.handle, ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetInMulti(self, startio, endio):
        '''
        :Description:读取多个输入信号。

        :param startio:IO 口起始编号。       type: int

        :param endio:IO 口结束编号。       type: int

        :Return:错误码,按位获取的输入口的状态值。最多存储32 个输出口状态。    type:int32,int32

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetInMulti(self.handle, ctypes.c_uint(startio), ctypes.c_uint(endio),
                                             ctypes.byref(value))
        return ret, value

    def ZAux_SetTimeOut(self, timems):
        '''
        :Description:命令的延时等待时间。

        :param timems:等待时间 MS。  type: int

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_SetTimeOut(self.handle, ctypes.c_uint(timems))
        return ret

    def ZAux_Direct_HwPswitch2(self, Axisnum, Mode, Opnum, Opstate, ModePara1, ModePara2, ModePara3, ModePara4):
        '''
        :Description:硬件位置比较输出2 4系列产品, 20170513以上版本支持.  ZMC306E/306N支持。

        :param Axisnum:比较输出的轴号。       type: int

        :param Mode:模式 1-启动比较器。       type: int
                         2- 停止并删除没完成的比较点。
                         3- 矢量比较方式。
                         4- 矢量比较方式, 单个比较点。
                         5- 矢量比较方式, 周期脉冲模式。
                         6-矢量比较方式, 周期模式, 这种模式一般与HW_TIMER一起使用

        :param Opnum:输出口编号。4 系列 out 0-为硬件位置比较输出。       type: int

        :param Opstate:第一个比较点的输出状态。 0-关闭 1-打开。         type: int

        :param ModePara1:多功能参数。       type: float

        :param ModePara2:多功能参数。       type: float

        :param ModePara3:多功能参数。       type: float

        :param ModePara4:多功能参数。       type: float

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_Direct_HwPswitch2(self.handle, ctypes.c_int(Axisnum), ctypes.c_int(Mode),
                                             ctypes.c_int(Opnum), ctypes.c_int(Opstate), ctypes.c_float(ModePara1),
                                             ctypes.c_float(ModePara2), ctypes.c_float(ModePara3),
                                             ctypes.c_float(ModePara4))
        return ret

    def ZAux_Direct_SetOutMulti(self, iofirst, ioend, istate):
        '''
        :Description:IO 设置路输出状态。

        :param iofirst:IO口起始编号。  type: int

        :param ioend:IO口结束编号。  type: int

        :param istate:。输出口状态,istate 按位设置，一个 UNIT对应 32 个输出口状态(列表类型)。  type: int32

        :Return:错误码,输出口状态。    type:int32

        '''
        arry = (ctypes.c_uint32 * len(istate))(*istate)
        ret = zauxdll.ZAux_Direct_SetOutMulti(self.handle, ctypes.c_uint(iofirst), ctypes.c_int(ioend), arry)
        return ret

    def ZAux_Direct_GetOutMulti(self, iofirst, ioend):
        '''
        :Description:IO 接口获取多路输出状态。

        :param iofirst:IO口起始编号。  type: int

        :param ioend:IO口结束编号。  type: int

        :Return:错误码,输出口状态。    type:int32

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetOutMulti(self.handle, ctypes.c_uint(iofirst), ctypes.c_int(ioend),
                                              ctypes.byref(value))
        return ret, value

    def ZAux_Direct_MultiMove(self, iMoveLen, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:多条相对多轴直线插补 。

        :param iMoveLen:填写的运动长度。  type: int

        :param imaxaxises:参与运动总轴数。  type: int

        :param piAxislist:轴号列表。  type: int

        :param pfDisancelist:距离列表。  type: float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(piAxislist))(*piAxislist)
        b = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_MultiMove(self.handle, iMoveLen, imaxaxises, value, b)
        return ret

    def ZAux_Direct_MultiMoveAbs(self, iMoveLen, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:多条绝对多轴直线插补 。

        :param iMoveLen:填写的运动长度。  type: int

        :param imaxaxises:参与运动总轴数。  type: int

        :param piAxislist:轴号列表。  type: int

        :param pfDisancelist:距离列表。  type: float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(piAxislist))(*piAxislist)
        b = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_MultiMoveAbs(self.handle, iMoveLen, imaxaxises, value, b)
        return ret

    def ZAux_Direct_MultiMovePt(self, iMoveLen, imaxaxises, piAxislist, pTickslist, pfDisancelist):
        '''
        :Description:多条相对PT运动 。

        :param iMoveLen:填写的运动数量。  type: int

        :param imaxaxises:参与运动总轴数。  type: int

        :param piAxislist:轴号列表。  type: int

        :param piAxislist:周期列表。  type: int

        :param pfDisancelist:距离列表。  type: float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(piAxislist))(*piAxislist)
        b = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        a = (ctypes.c_float * len(pTickslist))(*pTickslist)
        ret = zauxdll.ZAux_Direct_MultiMovePt(self.handle, iMoveLen, imaxaxises, value, a, b)
        return ret

    def ZAux_Direct_MultiMovePtAbs(self, iMoveLen, imaxaxises, piAxislist, pTickslist, pfDisancelist):
        '''
        :Description:多条绝对PT运动 。

        :param iMoveLen:填写的运动数量。  type: int

        :param imaxaxises:参与运动总轴数。  type: int

        :param piAxislist:轴号列表。  type: int

        :param piAxislist:周期列表。  type: int

        :param pfDisancelist:距离列表。  type: float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(piAxislist))(*piAxislist)
        b = (ctypes.c_float * len(pfDisancelist))(*pfDisancelist)
        a = (ctypes.c_float * len(pTickslist))(*pTickslist)
        ret = zauxdll.ZAux_Direct_MultiMovePtAbs(self.handle, iMoveLen, imaxaxises, value, a, b)
        return ret

    def ZAux_ZarDown(self, Filename, run_mode):
        '''
        :Description:下载 ZAR 程序到控制器运行。

        :param Filename:BAS 文件名带路径。  type: sting

        :param run_mode:下载模式 0-RAM 1-ROM。  type: int32

        :Return:错误码。    type:int32

        '''
        _str = Filename.encode('utf-8')
        ret = zauxdll.ZAux_ZarDown(self.handle, _str, ctypes.c_int32(run_mode))
        return ret

    def ZAux_SetRtcTime(self, RtcDate, RtcTime):
        '''
        :Description:设置RTC时间。

        :param RtcDate:系统日期 格式YYMMDD。  type: sting

        :param RtcTime:系统时间 格式HHMMSS。  type: sting

        :Return:错误码。    type:int32

        '''
        _STR = RtcDate.encode('utf-8')
        _STB = RtcTime.encode('utf-8')
        ret = zauxdll.ZAux_SetRtcTime(self.handle, _STR, _STB)
        return ret

    def ZAux_FastOpen(self, type, pconnectstring, uims):
        '''
        :Description:与控制器建立链接, 可以指定连接的等待时间。

        :param type:连接类型   1-COM 2-ETH 3-预留USB 4-PCI。  type: int

        :param pconnectstring:连接字符串 pconnectstring  COM口号/IP地址。  type: sting

        :param uims:连接超时时间 uims。  type:int

        :Return:错误码。    type:int32

        '''
        ip_bytes = pconnectstring.encode('utf-8')
        ret = zauxdll.ZAux_FastOpen(type, ip_bytes, uims, ctypes.pointer(self.handle))
        return ret

    def ZAux_Direct_UserDatum(self, iaxis, imode, HighSpeed, LowSpeed, DatumOffset):
        '''
        :Description:自定义二次回零。

        :param iaxis:轴号。  type: int

        :param imode:回零模式。  type: int

        :param HighSpeed:回零高速。  type:float

        :param LowSpeed:回零低速。  type:float

        :param DatumOffset:二次回零偏移距离。  type:float

        :Return:错误码。    type:int32

        '''
        ret = zauxdll.ZAux_Direct_UserDatum(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode),
                                            ctypes.c_float(HighSpeed), ctypes.c_float(LowSpeed),
                                            ctypes.c_float(DatumOffset))
        return ret

    def ZAux_Direct_Pitchset(self, iaxis, iEnable, StartPos, maxpoint, DisOne, TablNum, pfDisancelist):
        '''
        :Description:设置轴的螺距补偿，扩展轴无效。

        :param iaxis:轴号。  type: int

        :param iEnable:是否启用补偿。  type: int

        :param StartPos:起始补偿MPOS位置。  type:float

        :param maxpoint:补偿区间总点数。  type:uint32

        :param DisOne:每个补偿点间距。  type:float

        :param TablNum: 补偿坐标值填入TABLE系统数组起始引导地址。  type:uint32

        :param pfDisancelist:区间补偿值列表。  type:float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Pitchset(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iEnable),
                                           ctypes.c_float(StartPos), ctypes.c_int(maxpoint), ctypes.c_float(DisOne),
                                           ctypes.c_int(TablNum), value)
        return ret

    def ZAux_Direct_Pitchset2(self, iaxis, iEnable, StartPos, maxpoint, DisOne, TablNum, pfDisancelist, RevTablNum,
                              RevpfDisancelist):
        '''
        :Description:设置轴的螺距双向补偿，扩展轴无效。

        :param iaxis:轴号。  type: int

        :param iEnable:是否启用补偿。  type: int

        :param StartPos:起始补偿MPOS位置。  type:float

        :param maxpoint:补偿区间总点数。  type:uint32

        :param DisOne:每个补偿点间距。  type:float

        :param TablNum: 补偿坐标值填入TABLE系统数组起始引导地址。  type:uint32

        :param pfDisancelist:区间补偿值列表。  type:float

        :param RevTablNum:反向补偿坐标值填入TABLE系统数组起始引导地址。  type:uint32

        :param RevpfDisancelist:反向区间补偿值列表 补偿数据方向于正向方向一致。  type:float

        :Return:错误码。    type:int32

        '''
        value = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        b = (ctypes.c_int * len(RevpfDisancelist))(*RevpfDisancelist)
        ret = zauxdll.ZAux_Dire

    #
    # itchset2(self.handle, ctypes.c_int(iaxis), ctypes.c_int(iEnable),
    #          ctypes.c_float(StartPos), ctypes.c_int(maxpoint), ctypes.c_float(DisOne),
    #          ctypes.c_int(TablNum), value, ctypes.c_int(RevTablNum), b)
    #     return ret

#''''''
def ZAux_CycleUpEnable(self, cycleindex, fintervalms, psetesname):
    # '''
    # :Description:使能周期上报。
    #
    # :param cycleindex:报通道号, 0-最大值-1。  type: uint32
    # 
    # :param fintervalms:上报间隔时间, ms单位, 不能低于控制器SERVO_PERIOD。  type: int
    #
    # :param psetesname:上报参数选择, 语法: 参数1, 参数2(index), 参数3(index, numes)。  type:sting
    #
    # :Return:错误码。    type:int32
    #
    # '''
    _str = psetesname.encode('utf-8')
    ret = zauxdll.ZAux_CycleUpEnable(self.handle, ctypes.c_int(cycleindex), ctypes.c_float(fintervalms), _str)
    return ret


def ZAux_CycleUpDisable(self, cycleindex):
    # '''
    # :Description:去使能周期上报。
    #
    # :param cycleindex:报通道号, 0-最大值-1。  type: uint32
    #
    # :Return:错误码。    type:int32
    #
    # '''

    ret = zauxdll.ZAux_CycleUpDisable(self.handle, ctypes.c_int(cycleindex))
    return ret


def ZAux_CycleUpGetRecvTimes(self, cycleindex):
    '''
    :Description: 周期上报收到的包次数, 超过溢出. 调试使用。

    :param cycleindex:报通道号, 0-最大值-1。  type: uint32

    :Return:错误码。    type:int32

    '''

    ret = zauxdll.ZAux_CycleUpGetRecvTimes(self.handle, ctypes.c_int(cycleindex))
    return ret


def ZAux_CycleUpForceOnce(self, cycleindex):
    '''
    :Description: 强制上报一次, 在运动指令后idle可能不准确的情况下调用.。

    :param cycleindex:报通道号, 0-最大值-1。  type: uint32

    :Return:错误码。    type:int32

    '''
    ret = zauxdll.ZAux_CycleUpGetRecvTimes(self.handle, ctypes.c_int(cycleindex))
    return ret


def ZAux_CycleUpReadBuff(self, cycleindex, psetname, isetindex):
    '''
    :Description:从周期上报里面读取内容.。

    :param cycleindex: -1, 自动选择cycle编号。  type: uint32

    :param psetname:参数名称。  type:string

    :param isetindex:参数编号。  type: uint32

    :Return:错误码,返回值。    type:int32,double

    '''
    _str = psetname.encode('utf-8')
    value = (ctypes.c_double)()
    ret = zauxdll.ZAux_CycleUpReadBuff(self.handle, ctypes.c_int32(cycleindex), _str, ctypes.c_int32(isetindex),
                                       ctypes.byref(value))
    return ret, value


def ZAux_CycleUpReadBuffInt(self, cycleindex, psetname, isetindex):
    '''
    :Description:从周期上报里面读取内容。

    :param cycleindex: -1, 自动选择cycle编号。  type: uint32

    :param psetname:参数名称。  type:string

    :param isetindex:参数编号。  type: uint32

    :Return:错误码,返回值。    type:int32,int32

    '''
    _str = psetname.encode('utf-8')
    value = (ctypes.c_int32)()
    ret = zauxdll.ZAux_CycleUpReadBuffInt(self.handle, ctypes.c_int32(cycleindex), _str, ctypes.c_int32(isetindex),
                                          ctypes.byref(value))
    return ret, value


def ZAux_Direct_MultiLineN(self, imode, iMoveLen, imaxaxises, piAxislist, pfDisancelist):
    '''
    :Description:多轴多段线直线连续插补 。

    :param imode:bit0- bifabs
                 bit1- bifsp		是否
                 bit2- bifresume
                 bit3- bifmovescan 调用。  type: int

    :param iMoveLen:填写的运动长度。  type: int

    :param imaxaxises:参与运动总轴数。  type:int

    :param piAxislist:轴号列表。  type:uint32

    :param pfDisancelist:距离列表  iMoveLen * imaxaxises。  type:float

    :Return:错误码, 剩余缓冲可以下发的最大命令长度。    type:int32,int

    '''
    a = (ctypes.c_int * len(piAxislist))(*piAxislist)
    b = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
    ret = zauxdll.ZAux_Direct_MultiLineN(self.handle, ctypes.c_int(imode), ctypes.c_int(iMoveLen),
                                         ctypes.c_int(imaxaxises), a, b)
    return ret


def ZAux_Direct_MoveSync(self, imode, synctime, syncposition, syncaxis, imaxaxises, piAxislist, pfDisancelist):
    '''
    :Description:皮带同步跟随运动 。

    :param imode:同步模式 -1结束模式  -2强制结束 0-第一个轴跟随 10-第二个轴跟随 20-第二个轴跟随  小数位-angle:皮带旋转角度	type: float

    :param synctime:填写的运动长度。  type: int

    :param syncposition:同步时间,ms单位,本运动在指定时间内完成,完成时BASE轴跟上皮带且保持速度一致。0表示根据运动轴的速度加速度来估计同步时间,可能不准确  。  type:int

    :param syncaxis:皮带轴轴号。  type:uint32

    :param imaxaxises:参与同步从轴总数  。  type:int

    :param piAxislist:从站轴号列表。  type:int

    :param pfDisancelist:皮带轴物体被感应到时从轴的绝对坐标位置列表。  type:float

    :Return:错误码。    type:int32

    '''
    a = (ctypes.c_int * len(piAxislist))(*piAxislist)
    b = (ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
    ret = zauxdll.ZAux_Direct_MoveSync(self.handle, ctypes.c_int(imode), ctypes.c_int(synctime),
                                       ctypes.c_int(syncposition), ctypes.c_int(syncaxis), ctypes.c_int(imaxaxises),
                                       a, b)
    return ret


def ZAux_Direct_MoveCancel(self, base_axis, Cancel_Axis, iMode):
    '''
    :Description:运动缓冲区中取消其他轴运动，以达到运动中取消其他轴运动的效果。

    :param base_axis:运动轴轴号。	type: int32

    :param Cancel_Axis:停止轴轴号。  type: int32

    :param iMode:停止模式。  type:int
                0:取消当前运动
                1:取消缓冲的运动
                2:取消当前运动和缓冲的运动
                3:立即中断脉冲发送

    :Return:错误码。    type:int32

    '''
    ret = zauxdll.ZAux_Direct_MoveCancel(self.handle, ctypes.c_int(base_axis), ctypes.c_int(Cancel_Axis),
                                         ctypes.c_int(iMode))
    return ret


def ZAux_Direct_CycleRegist(self, iaxis, imode, iTabStart, iTabNum):
    '''
    :Description:连续位置锁存指令 。

    :param iaxis:轴号。	type: int

    :param imode:锁存模式。  type: int

    :param iTabStart:连续锁存的内容存储的table位置,第一个table元素存储锁存的个数,后面存储锁存的坐标,最多保存个数= numes-1,溢出时循环写入。  type:int

    :param iTabNum:占用的table个数。  type:int

    :Return:错误码。    type:int32

    '''
    ret = zauxdll.ZAux_Direct_CycleRegist(self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode),
                                          ctypes.c_int(iTabStart), ctypes.c_int(iTabNum))
    return ret


def ZAux_BusCmd_NodePdoWrite(self, inode, index, subindex, type, vvalue):
    '''
    :Description:Pdo写操作。

    :param inode:节点号。	type: int32

    :param index:对象字典。  type: int32

    :param subindex:子对象。  type:uint32

    :param type:数据类型。  type:uint32

    :param vvalue:写入数据值。  type:int

    :Return:错误码。    type:int32

    '''
    ret = zauxdll.ZAux_BusCmd_NodePdoWrite(self.handle, ctypes.c_int(inode), ctypes.c_int(index),
                                           ctypes.c_int(subindex), ctypes.c_int(type), ctypes.c_int(vvalue))
    return ret


def ZAux_BusCmd_NodePdoRead(self, inode, index, subindex, type):
    '''
    :Description:Pdo读操作。

    :param inode:节点号。	type: int32

    :param index:对象字典。  type: int32

    :param subindex:子对象。  type:uint32

    :param type:数据类型。  type:uint32

    :param vvalue:写入数据值。  type:int

    :Return:错误码,读取数据值。    type:int32,int

    '''
    value = (ctypes.c_int)()
    ret = zauxdll.ZAux_BusCmd_NodePdoRead(self.handle, ctypes.c_int(inode), ctypes.c_int(index),
    ctypes.c_int(subindex), ctypes.c_int(type), ctypes.byref(value))
    return ret, value

# #测试
# #
# Czauxdll=ZAUXDLL()
# Czauxdll.ZAux_OpenEth("192.168.0.22")
#
#
#
# ret=Czauxdll.ZAux_Modbus_Set0x(20000,4,[15])
#
# print(ret)


#
#


# for vr in ip:


#     print(vr)
