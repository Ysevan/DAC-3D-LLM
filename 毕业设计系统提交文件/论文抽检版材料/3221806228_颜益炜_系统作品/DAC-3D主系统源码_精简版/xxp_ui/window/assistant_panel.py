"""Stable embedded DAC-3D assistant dock for the xxp_ui main system."""

from __future__ import annotations

import html
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtWidgets


XXP_UI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSISTANT_ROOT = PROJECT_ROOT / "dac3d_iim_assistant"
if str(ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSISTANT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOWS_PATH_PATTERN = re.compile(r"(?P<path>[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*)")


class DAC3DMainWindowBridge:
    """Bridge assistant commands to the live xxp_ui detection queues."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self.window = window

    def start_online_scan(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        del command
        self.window.startDetectionRun()
        self.window.ui_debug_queue.put({"type": "Control", "func": "Scan"})
        return {
            "accepted": True,
            "status": self.query_current_status(),
            "message": "已向 DAC-3D 主系统发送在线扫描指令。",
        }

    def start_offline_detection(self, command: dict[str, Any]) -> dict[str, Any]:
        payload = dict(command.get("payload") or {})
        image_folder = str(payload.get("image_folder") or "")
        validation = self._validate_offline_folder(image_folder)
        if not validation["ready"]:
            return {
                "accepted": False,
                "status": self.query_current_status(),
                "validation": validation,
                "message": "离线目录校验未通过，未启动检测。",
            }

        self.window.offline_source_dir = image_folder
        if hasattr(self.window, "offlineCheckBox"):
            self.window.offlineCheckBox.setChecked(True)
        self.window.startDetectionRun()
        self.window.ui_image_queue.put(
            {
                "type": "offline_detect_folder",
                "data": {
                    "folder": image_folder,
                    "tray_id": getattr(self.window, "Tray_id", None),
                },
            }
        )
        return {
            "accepted": True,
            "status": self.query_current_status(),
            "validation": validation,
            "message": "已向 DAC-3D 主系统发送离线检测指令。",
        }

    def stop_detection(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        del command
        self.window.stop()
        return {
            "accepted": True,
            "status": self.query_current_status(),
            "message": "已向 DAC-3D 主系统发送停止检测指令。",
        }

    def query_current_status(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        del command
        running = not bool(self.window.ui.runBtn.isEnabled())
        sample_count = len(getattr(self.window, "samples_result", []) or [])
        progress = min(100, int(sample_count / 144 * 100)) if running else 0
        state = "running" if running else "idle"
        if getattr(self.window, "stopFlag", False):
            state = "stopped"
        return {
            "state": state,
            "progress": progress,
            "message": f"主系统当前记录样品数：{sample_count}/144。",
            "tray_id": getattr(self.window, "Tray_id", None),
            "offline_mode": bool(
                hasattr(self.window, "offlineCheckBox") and self.window.offlineCheckBox.isChecked()
            ),
            "offline_source_dir": getattr(self.window, "offline_source_dir", ""),
        }

    def validate_offline_folder(self, command: dict[str, Any]) -> dict[str, Any]:
        payload = dict(command.get("payload") or {})
        validation = self._validate_offline_folder(str(payload.get("image_folder") or ""))
        return {
            "accepted": validation["ready"],
            "status": self.query_current_status(),
            "validation": validation,
            "message": "离线目录可用于检测。" if validation["ready"] else "离线目录不完整。",
        }

    def get_latest_result_summary(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        del command
        result_root = XXP_UI_ROOT / "runtime" / "ftkpic" / "results"
        latest_dir = None
        if result_root.exists():
            candidates = [path for path in result_root.iterdir() if path.is_dir()]
            if candidates:
                latest_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        files: list[str] = []
        if latest_dir is not None:
            files = [str(path) for path in sorted(latest_dir.rglob("*")) if path.is_file()][-20:]
        return {
            "result_root": str(latest_dir or result_root),
            "files": files,
            "parsed_result": self.get_recent_inspection_result(),
            "status": self.query_current_status(),
            "message": "已读取最近一次检测结果目录。" if files else "未找到检测结果文件。",
        }

    def get_recent_inspection_result(self) -> dict[str, Any]:
        samples = list(getattr(self.window, "samples_result", []) or [])
        latest = samples[-1] if samples else {}
        return {
            "defect_type": latest.get("type", "unknown") if isinstance(latest, dict) else "unknown",
            "location": latest.get("position", "latest sample") if isinstance(latest, dict) else "latest sample",
            "confidence": latest.get("confidence", 0.0) if isinstance(latest, dict) else 0.0,
            "measurements": latest.get("measurements", {}) if isinstance(latest, dict) else {},
            "sample_id": latest.get("sample_id") if isinstance(latest, dict) else None,
        }

    def runtime_snapshot(self) -> dict[str, Any]:
        return {
            "host": "xxp_ui",
            "control": "ui_debug_queue/ui_image_queue",
            "status": self.query_current_status(),
        }

    def _validate_offline_folder(self, image_folder: str) -> dict[str, Any]:
        path = Path(image_folder)
        result = {
            "image_folder": image_folder,
            "exists": path.exists(),
            "checked_files": 0,
            "missing_requirements": [],
            "ready": False,
        }
        if not image_folder:
            result["missing_requirements"].append("image_folder")
            return result
        if not path.exists() or not path.is_dir():
            result["missing_requirements"].append("existing_directory")
            return result
        image_files = [file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES]
        result["checked_files"] = len(image_files)
        if not image_files:
            result["missing_requirements"].append("image_files")
        result["ready"] = not result["missing_requirements"]
        return result


class AssistantWorker(QtCore.QThread):
    """Run non-control assistant calls away from the PyQt UI thread."""

    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, assistant: Any, message: str, history: list[tuple[str, str]]) -> None:
        super().__init__()
        self.assistant = assistant
        self.message = message
        self.history = history

    def run(self) -> None:
        try:
            self.succeeded.emit(self.assistant.handle_message(self.message, self.history))
        except BaseException:
            self.failed.emit(traceback.format_exc())


class AssistantDock(QtWidgets.QDockWidget):
    """Modern, safe assistant panel embedded in the DAC-3D main window."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__("DAC-3D 智能助手", window)
        self.window = window
        self.bridge = DAC3DMainWindowBridge(window)
        self.history: list[tuple[str, str]] = []
        self.assistant: Any | None = None
        self.worker: AssistantWorker | None = None
        self._build_ui()
        self._load_assistant()

    def _build_ui(self) -> None:
        self.setObjectName("dac3dAssistantDock")
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.setMinimumWidth(420)

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("assistantHeader")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        title = QtWidgets.QLabel("DAC-3D IIM 智能助手")
        title.setObjectName("assistantTitle")
        subtitle = QtWidgets.QLabel("嵌入主系统：状态查询、检测控制、结果读取、知识库问答。")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("assistantSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        self.output = QtWidgets.QTextBrowser()
        self.output.setObjectName("assistantOutput")
        self.output.setOpenExternalLinks(False)

        self.input = QtWidgets.QLineEdit()
        self.input.setObjectName("assistantInput")
        self.input.setPlaceholderText("输入问题或指令，例如：当前检测状态 / 停止检测")
        self.input.returnPressed.connect(self._send)

        send_button = QtWidgets.QPushButton("发送")
        send_button.clicked.connect(self._send)
        self.status_button = QtWidgets.QPushButton("状态")
        self.status_button.clicked.connect(lambda: self._send_text("当前检测状态"))
        self.latest_button = QtWidgets.QPushButton("结果")
        self.latest_button.clicked.connect(lambda: self._send_text("读取最新检测结果"))
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(lambda: self._send_text("停止检测"))
        clear_button = QtWidgets.QPushButton("清空")
        clear_button.clicked.connect(self._clear)

        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(self.input, 1)
        input_row.addWidget(send_button)
        quick_row = QtWidgets.QHBoxLayout()
        quick_row.addWidget(self.status_button)
        quick_row.addWidget(self.latest_button)
        quick_row.addWidget(self.stop_button)
        quick_row.addWidget(clear_button)

        layout.addWidget(header)
        layout.addWidget(self.output, 1)
        layout.addLayout(input_row)
        layout.addLayout(quick_row)
        self.setWidget(root)
        self.setStyleSheet(self._style_sheet())
        self._append_assistant(
            "助手已就绪。建议演示指令：当前检测状态、读取最新检测结果、停止检测、"
            "validate offline folder C:\\...。"
        )

    def _style_sheet(self) -> str:
        return """
        QDockWidget#dac3dAssistantDock {
            background: #eef6ff;
            border: 1px solid #bdd8ff;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }
        QFrame#assistantHeader {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #e8f3ff);
            border: 1px solid #cfe1fa;
            border-radius: 18px;
        }
        QLabel#assistantTitle {
            color: #0d2f5f;
            font-size: 19px;
            font-weight: 800;
        }
        QLabel#assistantSubtitle {
            color: #4f6682;
            font-size: 12px;
        }
        QTextBrowser#assistantOutput {
            background: #ffffff;
            color: #17263a;
            border: 1px solid #d5e5fa;
            border-radius: 18px;
            padding: 12px;
            font-size: 13px;
        }
        QLineEdit#assistantInput {
            background: #ffffff;
            border: 1px solid #b8d1f4;
            border-radius: 14px;
            padding: 10px 12px;
            color: #17263a;
        }
        QPushButton {
            background: #1677ff;
            color: white;
            border: 0;
            border-radius: 12px;
            padding: 9px 13px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #0b63dc;
        }
        QPushButton:disabled {
            background: #9db7d9;
        }
        """

    def _load_assistant(self) -> None:
        try:
            from app import DAC3DAssistant
            from config import AppConfig

            config = AppConfig.from_env(ASSISTANT_ROOT)
            config.mock_mode = False
            self.assistant = DAC3DAssistant.create(config, runtime_bridge=self.bridge)
            self._append_system("知识库助手已加载。")
        except BaseException:
            self.assistant = None
            self._append_system("知识库助手加载失败，控制类指令仍可使用。")
            self._append_error(traceback.format_exc())

    def _send_text(self, text: str) -> None:
        self.input.setText(text)
        self._send()

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append_user(text)
        handled = self._handle_embedded_command(text)
        if handled:
            return
        if self.assistant is None:
            self._append_assistant("知识库助手未加载成功。请先使用状态、结果、停止等主系统控制指令。")
            return
        self._set_busy(True)
        self.worker = AssistantWorker(self.assistant, text, list(self.history))
        self.worker.succeeded.connect(lambda response: self._on_worker_succeeded(text, response))
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _handle_embedded_command(self, text: str) -> bool:
        lowered = text.lower()
        try:
            if self._contains_any(lowered, ("状态", "status", "进度")):
                self._render_result("当前状态", self.bridge.query_current_status())
                return True
            if self._contains_any(lowered, ("停止", "stop", "abort", "cancel")):
                self._render_result("停止检测", self.bridge.stop_detection())
                return True
            if self._contains_any(lowered, ("最新结果", "读取结果", "latest result", "get latest", "检测结果")):
                self._render_result("最新结果", self.bridge.get_latest_result_summary())
                return True
            if self._contains_any(lowered, ("validate offline folder", "校验离线", "检查离线", "离线目录")):
                image_folder = self._extract_windows_path(text)
                command = {"payload": {"image_folder": image_folder}}
                if self._contains_any(lowered, ("开始", "启动", "execute", "run", "start offline")):
                    self._render_result("离线检测", self.bridge.start_offline_detection(command))
                else:
                    self._render_result("离线目录校验", self.bridge.validate_offline_folder(command))
                return True
            if self._contains_any(lowered, ("start online", "online scan", "在线扫描", "在线检测")):
                if self._contains_any(lowered, ("execute", "执行", "开始", "启动")):
                    self._render_result("在线扫描", self.bridge.start_online_scan())
                else:
                    self._append_assistant("在线扫描属于真实控制指令。请明确输入“开始在线扫描”或“start online scan execute”。")
                return True
        except BaseException:
            self._append_assistant("主系统控制指令执行失败，错误已捕获，主程序不会退出。")
            self._append_error(traceback.format_exc())
            return True
        return False

    def _on_worker_succeeded(self, user_text: str, response: Any) -> None:
        answer = str(getattr(response, "answer", "") or "已完成。")
        self.history.append((user_text, answer))
        self._append_assistant(answer)
        command_preview = getattr(response, "command_preview", None)
        status_summary = getattr(response, "status_summary", None)
        parsed_result = getattr(response, "parsed_result", None)
        if command_preview:
            self._render_json("结构化命令", command_preview)
        if status_summary:
            self._render_json("运行状态", status_summary)
        if parsed_result:
            self._render_json("检测结果", parsed_result)

    def _on_worker_failed(self, error_text: str) -> None:
        self._append_assistant("助手处理失败，错误已捕获，主程序不会退出。")
        self._append_error(error_text)

    def _render_result(self, title: str, result: dict[str, Any]) -> None:
        message = str(result.get("message") or title)
        self._append_assistant(message)
        self._render_json(title, result)

    def _render_json(self, title: str, value: Any) -> None:
        escaped = html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        self.output.append(f"<div class='card'><b>{html.escape(title)}</b><pre>{escaped}</pre></div>")

    def _append_user(self, text: str) -> None:
        self.output.append(f"<p style='color:#0d47a1'><b>操作员：</b>{html.escape(text)}</p>")

    def _append_assistant(self, text: str) -> None:
        self.output.append(f"<p><b>助手：</b>{html.escape(text)}</p>")

    def _append_system(self, text: str) -> None:
        self.output.append(f"<p style='color:#58708c'><b>系统：</b>{html.escape(text)}</p>")

    def _append_error(self, text: str) -> None:
        escaped = html.escape(text[-3000:])
        self.output.append(f"<details><summary>错误详情</summary><pre>{escaped}</pre></details>")

    def _set_busy(self, busy: bool) -> None:
        self.input.setEnabled(not busy)
        self.status_button.setEnabled(not busy)
        self.latest_button.setEnabled(not busy)
        self.stop_button.setEnabled(not busy)
        if busy:
            self._append_system("正在后台处理...")

    def _clear(self) -> None:
        self.history.clear()
        self.output.clear()
        self._append_assistant("上下文已清空。")

    def _extract_windows_path(self, text: str) -> str:
        match = WINDOWS_PATH_PATTERN.search(text)
        if not match:
            return ""
        path = match.group("path").rstrip("。；;，,")
        for suffix in (" execute", " submit", " run now", " start now"):
            if path.lower().endswith(suffix):
                path = path[: -len(suffix)].rstrip()
        return path

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in text for keyword in keywords)


def install_assistant_dock(window: QtWidgets.QMainWindow) -> AssistantDock:
    """Install the assistant dock into the live DAC-3D main window."""
    dock = AssistantDock(window)
    window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    return dock
