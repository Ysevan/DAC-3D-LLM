import sys
import numpy as np
from PyQt5.QtCore import (Qt, pyqtSignal, QPoint, QPropertyAnimation,
                          QEasingCurve, QRectF, QEvent)
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QColor, QFont,
                         QLinearGradient, QBrush, QPalette, QMouseEvent, QTransform)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QSizePolicy, QFrame, QStackedLayout, QScrollBar)


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.fullscreen_window = None

    def setup_ui(self):
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setStyleSheet("background: #1E1E1E; border-radius: 8px;")

    def display_image(self, source):
        pixmap = self.create_pixmap(source)
        if pixmap.isNull():
            return

        if hasattr(self, 'image_item'):
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
        self.fit_view()

    def create_pixmap(self, source):
        if isinstance(source, str):
            return QPixmap(source)
        elif isinstance(source, np.ndarray):
            return self.numpy_to_pixmap(source)
        return QPixmap()

    def numpy_to_pixmap(self, arr):
        height, width = arr.shape[0], arr.shape[1]
        if arr.ndim == 2:
            return QPixmap.fromImage(QImage(arr.data, width, height, width, QImage.Format_Grayscale8))
        elif arr.ndim == 3:
            channels = arr.shape[2]
            fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
            return QPixmap.fromImage(QImage(arr.data, width, height, width * channels, fmt).rgbSwapped())
        return QPixmap()

    def fit_view(self):
        if hasattr(self, 'image_item'):
            self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_view()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_fullscreen()
        super().mousePressEvent(event)

    def show_fullscreen(self):
        if hasattr(self, 'image_item'):
            self.fullscreen_window = FullscreenViewer(self.image_item.pixmap())
            self.fullscreen_window.closed.connect(self.handle_fullscreen_close)
            self.fullscreen_window.showFullScreen()

    def handle_fullscreen_close(self):
        self.fullscreen_window = None


class FullscreenViewer(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.setup_ui()
        self.setup_animations()
        self.adjust_initial_zoom()

    def setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        self.main_layout = QStackedLayout(main_widget)
        self.main_layout.setStackingMode(QStackedLayout.StackAll)

        # Content layer
        self.content_widget = QWidget()
        self.setup_content_ui()
        self.main_layout.addWidget(self.content_widget)

        # Mask layer
        self.mask_layer = QWidget()
        self.mask_layer.setStyleSheet("background-color: rgba(0, 0, 0, 0.85);")
        self.main_layout.addWidget(self.mask_layer)

    def setup_content_ui(self):
        # 主视图布局
        self.content_widget.setLayout(QVBoxLayout())
        self.content_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.content_widget.layout().setSpacing(0)

        # 创建图形视图
        self.view = GraphicsView()
        self.view.setScene(QGraphicsScene(self))
        self.image_item = self.view.scene().addPixmap(self.pixmap)
        self.image_item.setTransformationMode(Qt.SmoothTransformation)
        self.view.zoom_updated.connect(self.handle_zoom_updated)
        self.content_widget.layout().addWidget(self.view)

        # 创建悬浮工具栏
        self.create_floating_toolbar()

    def create_floating_toolbar(self):
        # 创建悬浮工具栏（修复版）
        self.toolbar = QWidget(self.content_widget)
        self.toolbar.setFixedHeight(48)
        self.toolbar.setFixedWidth(1500)

        # 修复样式表
        self.toolbar.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 0.95);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                margin: 0;
            }
        """)
        self.toolbar.setAttribute(Qt.WA_TranslucentBackground)

        # 修复布局边距
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(16, 6, 16, 6)  # 调整内边距
        toolbar_layout.setSpacing(20)

        # 添加弹性间隔保证元素居中
        toolbar_layout.addStretch()

        # 缩放标签
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("""
            QLabel {
                color: rgba(255,255,255,0.9);
                font: 500 14px 'Segoe UI';
                min-width: 80px;
                padding: 4px 12px;
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 4px;
            }
        """)

        # 关闭按钮
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 36)
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.12);
                border-radius: 18px;
                color: rgba(255,255,255,0.9);
                font: bold 16px 'Arial';
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.18);
            }
            QPushButton:pressed {
                background-color: rgba(255,255,255,0.08);
            }
        """)

        toolbar_layout.addWidget(self.zoom_label)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.close_btn)

        # 修复初始定位
        self.toolbar.setGeometry(
            self.content_widget.width() - 420,  # 屏幕宽度 - 工具栏宽度(400) - 右边距(20)
            20,
            400,
            48
        )
        self.toolbar.raise_()

    def resizeEvent(self, event):
        if hasattr(self, 'toolbar'):
            self.toolbar.setGeometry(
                self.content_widget.width() - 420,
                20,
                400,
                48
            )
        super().resizeEvent(event)
    def update_toolbar_position(self):
        self.toolbar.move(20, 20)
        self.toolbar.resize(self.content_widget.width() - 40, 48)


    def setup_animations(self):
        self.zoom_animation = QPropertyAnimation(self.zoom_label, b"windowOpacity")
        self.zoom_animation.setDuration(1200)
        self.zoom_animation.setStartValue(1.0)
        self.zoom_animation.setEndValue(0.0)
        self.zoom_animation.setEasingCurve(QEasingCurve.OutCubic)

    def adjust_initial_zoom(self):
        scale = 0.8
        center = self.view.mapToScene(self.view.rect().center())
        self.view.apply_zoom(scale, center)
        self.handle_zoom_updated(self.view.scale_factor)

    def handle_zoom_updated(self, factor):
        self.zoom_label.setText(f"{factor * 100:.1f}%")
        self.zoom_label.setWindowOpacity(1.0)
        self.zoom_animation.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class GraphicsView(QGraphicsView):
    zoom_updated = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.scale_factor = 1.0
        self.last_mouse_pos = None
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    def setup_ui(self):
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform |
                            QPainter.TextAntialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setStyleSheet("background: transparent;")
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def wheelEvent(self, event):
        mouse_pos = event.pos()
        scene_pos = self.mapToScene(mouse_pos)
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.apply_zoom(zoom_factor, scene_pos)

    def apply_zoom(self, factor, center):
        old_transform = self.transform()
        new_transform = QTransform()
        new_transform.translate(center.x(), center.y())
        new_transform.scale(factor, factor)
        new_transform.translate(-center.x(), -center.y())
        new_transform = new_transform * old_transform
        self.setTransform(new_transform)
        self.scale_factor *= factor
        self.zoom_updated.emit(self.scale_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_sample_image()

    def setup_ui(self):
        self.setWindowTitle("Professional Image Viewer")
        self.setGeometry(100, 100, 1024, 768)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer)

    def load_sample_image(self):
        self.viewer.display_image(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\1.bmp')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())