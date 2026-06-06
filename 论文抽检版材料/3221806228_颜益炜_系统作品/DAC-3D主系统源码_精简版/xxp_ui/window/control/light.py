import sys
import serial

class lightControllor:
    def __init__(self):
        self.light = None
        self.laser = None
        self.initSerialPort()

    def initSerialPort(self):
            #初始化串口
        self.light = serial.Serial(
            port='COM3',  # 串口号，根据实际情况修改
            baudrate=9600,  # 波特率
            bytesize=8,  # 数据位
            parity='N',  # 校验位
            stopbits=1,  # 停止位
            timeout=1  # 超时时间
        )
        # self.laser = serial.Serial(
        #     port='COM4',  # Serial port, adjust based on your system
        #     baudrate=9600,  # Baudrate
        #     bytesize=8,  # Data bits
        #     parity='N',  # Parity bit
        #     stopbits=1,  # Stop bit+
        #     timeout=1  # Timeout
        # )
    def light15control(self,mode):
        if mode == 0:
            command = "$20100027"
        else:
            command = "$10100024"
        self.light.write(command.encode('utf-8'))

    def light60control(self, mode):
        if mode == 0:
            command = "$20300025"
        else:
            command = '$10300026'
        self.light.write(command.encode('utf-8'))
# 激光
    def calculate_checksum(self, command):
            """计算校验位（校验和）"""
            return sum(command) & 0xFF  # 计算前 8 位字节的和并取低 8 位
    def generate_command(self, spinbox_value):
        """根据 doubleSpinBox_15 的值生成命令"""
        # 原始命令前 8 位
        base_command = [0x53, 0x0A, 0x02, 0x01, 0x00, 0x00, 0x00]

        # 将 spinbox_value 转换为 8 位字节（此处假设是整数）
        eighth_byte = spinbox_value & 0xFF  # 获取低 8 位

        # 添加第 8 位到命令中
        command = base_command + [eighth_byte]

        # 计算校验位（第 9 位）
        checksum = self.calculate_checksum(command)

        # 将校验位和最后一位（0x0D）添加到命令中
        command.append(checksum)
        command.append(0x0D)

        return command

    def laserpower(self, mode):
        if mode == 0:
            hex_command = bytes([0x53, 0x0A, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x5F, 0x0D])
            print('激光开')
        else:
            hex_command = bytes([0x53, 0x0A, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x5E, 0x0D])
            print('激光关')

        self.laser.write(hex_command)


    def laserpower1(self, spinbox_value):
        command = self.generate_command(spinbox_value)
        print("Generated Command:", [hex(byte) for byte in command])  # 打印生成的命令（以十六进制表示）
            # 发送命令到串口
        self.laser.write(bytes(command))

        # command = "$10100024"
        # command1= "$20100027"
        # command2='$10300026'
        # command4="$10300026"
        # light.write(command1.encode('utf-8'))

if __name__ == '__main__':
    lit = lightControllor()
    lit.laserpower1(1)