from PyQt5.QtWidgets import QLabel

class NotificationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            # print(f"创建 NotificationManager 实例，ID: {id(cls._instance)}")
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'notification_label'):  # 避免重复初始化
            self.notification_label = None

    def set_notification_label(self, label: QLabel):
        """设置通知栏的 QLabel"""
        self.notification_label = label
        # print(f"设置 notification_label: {self.notification_label}, 实例 ID: {id(self)}")

    def show_notification(self, message: str):
        """显示通知"""
        print(f"当前 notification_label: {self.notification_label}, 实例 ID: {id(self)}")
        if self.notification_label:
            self.notification_label.setText(message)
        else:
            print("Notification label is not set!")

notification_manager = NotificationManager()