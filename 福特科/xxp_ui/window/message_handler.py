from window.NotificationManager import notification_manager

import json
import socket
from PyQt5.QtCore import QThread, QObject, pyqtSignal


class ClientThread(QThread):
    def __init__(self, tcp_client):
        super().__init__()
        self.tcp_client = tcp_client

    def run(self):
        self.tcp_client.listen_for_messages()


class ClientReconnectThread(QThread):
    def __init__(self, tcp_client):
        super().__init__()
        self.tcp_client = tcp_client

    def run(self):
        while True:
            print("尝试重连服务器...")
            notification_manager.show_notification("尝试重连服务器...")
            res = self.tcp_client.connect_to_server()
            if res:
                print("重连成功！")
                notification_manager.show_notification("服务器重连成功！")
                break
            self.msleep(5000)  # 等待 5 秒后重试


class TCPClient(QObject):
    # 定义信号
    message_received = pyqtSignal(str)
    connection_status_changed = pyqtSignal(bool)  # 连接状态变化信号

    def __init__(self, host='localhost', port=27015, func=None):
        super().__init__()
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False  # 连接状态
        self.tcp_client_thread = None
        self.tcp_client_reconnect_thread = None
        self.func = func

        # 初始化时尝试连接服务器
        self.connect_to_server()

    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.running = True
            self.connection_status_changed.emit(True)  # 发送连接成功信号
            print("Connected to server.")
            notification_manager.show_notification("服务器连接成功！")
            self.start_listening()  # 启动消息监听线程
            if self.func:
                self.message_received.connect(self.func)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            # notification_manager.show_notification("服务器断开连接！")
            self.connection_status_changed.emit(False)  # 发送连接失败信号
            self.start_reconnect()  # 启动重连线程
            return False

    def start_listening(self):
        """启动消息监听线程"""
        if self.tcp_client_thread and self.tcp_client_thread.isRunning():
            self.tcp_client_thread.quit()  # 停止旧的线程
        self.tcp_client_thread = ClientThread(self)
        self.tcp_client_thread.start()

    def send_message(self, message_dict):
        if not self.running:
            return
        try:
            message_json = json.dumps(message_dict)
            self.client_socket.sendall(message_json.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send message: {e}")
            self.handle_disconnect()

    def listen_for_messages(self):
        while self.running:
            try:
                data = self.client_socket.recv(1024*1024*20).decode('utf-8')
                if not data:
                    break
                message_dict = json.loads(data)
                self.message_received.emit(json.dumps(message_dict))
            except Exception as e:
                print(f"Error receiving data: {e}")
                break
        self.handle_disconnect()

    def handle_disconnect(self):
        """处理断开连接"""
        self.running = False
        self.connection_status_changed.emit(False)  # 发送断开连接信号
        print("Disconnected from server.")
        self.start_reconnect()  # 启动重连线程

    def start_reconnect(self):
        """启动重连线程"""
        if self.tcp_client_reconnect_thread and self.tcp_client_reconnect_thread.isRunning():
            return  # 避免重复启动
        self.tcp_client_reconnect_thread = ClientReconnectThread(self)
        self.tcp_client_reconnect_thread.start()

    def disconnect_from_server(self):
        """主动断开连接"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.tcp_client_thread and self.tcp_client_thread.isRunning():
            self.tcp_client_thread.quit()
        if self.tcp_client_reconnect_thread and self.tcp_client_reconnect_thread.isRunning():
            self.tcp_client_reconnect_thread.quit()
        self.connection_status_changed.emit(False)  # 发送断开连接信号


from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import pyqtSlot

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("TCP Client Example")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        self.status_label = QLabel("状态: 未连接", self)
        layout.addWidget(self.status_label)

        self.tcp_client = TCPClient(func=self.handle_message)
        self.tcp_client.connection_status_changed.connect(self.update_status)

        connect_button = QPushButton("连接服务器", self)
        connect_button.clicked.connect(self.tcp_client.connect_to_server)
        layout.addWidget(connect_button)

        disconnect_button = QPushButton("断开连接", self)
        disconnect_button.clicked.connect(self.tcp_client.disconnect_from_server)
        layout.addWidget(disconnect_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    @pyqtSlot(bool)
    def update_status(self, connected):
        """更新连接状态"""
        if connected:
            self.status_label.setText("状态: 已连接")
        else:
            self.status_label.setText("状态: 未连接")

    @pyqtSlot(str)
    def handle_message(self, message):
        """处理接收到的消息"""
        print(f"收到消息: {message}")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()