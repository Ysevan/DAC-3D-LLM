import multiprocessing
import sys

from PyQt5.QtWidgets import QApplication
from multiprocessing import Process, Queue

from image_processor_22 import ImageProcessor
from window.login import Login
from window.debug_liubo_all  import Debug_UI


def openWindow(ui_debug_queue, debug_ui_queue,ui_image_queue,image_ui_queue,exit_event):
    app = QApplication(sys.argv)
    login = Login(ui_debug_queue, debug_ui_queue,ui_image_queue,image_ui_queue,exit_event)

    sys.exit(app.exec_())

def openDebug(debug_ui_queue,ui_debug_queue,debug_image_queue,image_debug_queue):
    app = QApplication(sys.argv)
    debug_UI = Debug_UI(debug_ui_queue,ui_debug_queue,debug_image_queue,image_debug_queue,mode='mainUICall', offline_hardware=True)
    app.exec_()

def open_image_processor(debug_image_queue, image_debug_queue,ui_image_queue,image_ui_queue):
    ImageProcessor(debug_image_queue, image_debug_queue,ui_image_queue,image_ui_queue)





if __name__ == '__main__':
    exit_event = multiprocessing.Event()

    ui_debug_queue = Queue()
    debug_ui_queue = Queue()

    debug_image_queue = Queue()
    image_debug_queue = Queue()

    ui_image_queue = Queue()
    image_ui_queue = Queue()



    # 启动图像处理进程
    processor = Process(target=open_image_processor, args=(debug_image_queue, image_debug_queue,ui_image_queue,image_ui_queue))
    processor.start()



    ui = Process(target=openWindow, args=(ui_debug_queue, debug_ui_queue,ui_image_queue,image_ui_queue,exit_event))
    ui.start()
    #
    debug = Process(target=openDebug, args=(debug_ui_queue,ui_debug_queue,debug_image_queue,image_debug_queue))
    debug.start()
    #
    exit_event.wait()

    ui.terminate()
    debug.terminate()
    processor.terminate()

