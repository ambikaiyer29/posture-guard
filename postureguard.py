"""
PostureGuard - macOS Posture Monitor with GUI
Uses MediaPipe PoseLandmarker via webcam to detect slouching.
Features: live camera preview, visual calibration, macOS notifications, slouch tracking.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QTabWidget,
    QFrame, QProgressBar, QScrollArea, QSystemTrayIcon, QMenu,
    QSizePolicy, QSpacerItem, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPainter, QPen, QAction


# ─────────────────────────────────────────────
# Model Download
# ─────────────────────────────────────────────

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILENAME = "pose_landmarker_lite.task"
DATA_DIR = os.path.expanduser("~/.postureguard")


def ensure_model() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(DATA_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("📥 Downloading pose model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"✅ Model saved to {model_path}")
    return model_path


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class SlouchEvent:
    timestamp: str
    duration_seconds: float

    @property
    def formatted_duration(self) -> str:
        m, s = divmod(int(self.duration_seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"


@dataclass
class Preferences:
    enable_notification: bool = True
    enable_sound: bool = True
    sensitivity: float = 0.06
    cooldown_seconds: int = 30

    def save(self):
        with open(os.path.join(DATA_DIR, "prefs.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> Preferences:
        try:
            with open(os.path.join(DATA_DIR, "prefs.json")) as f:
                return cls(**json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()


# ─────────────────────────────────────────────
# Posture Logger
# ─────────────────────────────────────────────

class PostureLogger:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.log_path = os.path.join(DATA_DIR, "posture_log.json")
        self.events: list[SlouchEvent] = self._load()

    def log(self, duration: float):
        if duration < 3.0:
            return
        self.events.append(SlouchEvent(
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
        ))
        self._save()

    def today_stats(self) -> tuple[int, float]:
        today = datetime.now().date()
        today_events = [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == today]
        return len(today_events), sum(e.duration_seconds for e in today_events)

    def weekly_stats(self) -> list[tuple[str, int, float]]:
        results = []
        for i in range(6, -1, -1):
            day = datetime.now().date() - timedelta(days=i)
            day_events = [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == day]
            results.append((day.strftime("%a"), len(day_events), sum(e.duration_seconds for e in day_events)))
        return results

    def today_events(self) -> list[SlouchEvent]:
        today = datetime.now().date()
        return [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == today]

    def clear(self):
        self.events.clear()
        self._save()

    def _save(self):
        cutoff = datetime.now() - timedelta(days=30)
        self.events = [e for e in self.events if datetime.fromisoformat(e.timestamp) > cutoff]
        with open(self.log_path, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)

    def _load(self) -> list[SlouchEvent]:
        try:
            with open(self.log_path) as f:
                return [SlouchEvent(**d) for d in json.load(f)]
        except (FileNotFoundError, json.JSONDecodeError):
            return []


# ─────────────────────────────────────────────
# Signal Bridge (thread-safe updates to GUI)
# ─────────────────────────────────────────────

class MonitorSignals(QObject):
    state_changed = pyqtSignal(str, float)       # state, deviation
    frame_ready = pyqtSignal(np.ndarray)          # BGR frame for preview
    calibration_progress = pyqtSignal(int, int)   # current, total
    log_message = pyqtSignal(str)                 # log text


# ─────────────────────────────────────────────
# Posture Monitor
# ─────────────────────────────────────────────

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class PostureMonitor:
    def __init__(self, prefs: Preferences, logger: PostureLogger, model_path: str):
        self.prefs = prefs
        self.logger = logger
        self.model_path = model_path
        self.signals = MonitorSignals()

        self.running = False
        self.calibrated = False
        self.state = "inactive"
        self.current_deviation = 0.0

        self.calibration_samples: list[float] = []
        self.baseline_ratio = 0.0
        self.calibration_count = 20

        self.slouch_start: Optional[float] = None
        self.last_alert_time: float = 0

        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.calibrated = False
        self.calibration_samples.clear()
        self._set_state("calibrating")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self.slouch_start is not None:
            self.logger.log(time.time() - self.slouch_start)
            self.slouch_start = None
        self._set_state("inactive")

    def recalibrate(self):
        self.calibrated = False
        self.calibration_samples.clear()
        self._set_state("calibrating")

    def _run_loop(self):
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

        self.signals.log_message.emit("📷 Opening camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.signals.log_message.emit("❌ Cannot open camera!")
            self._set_state("inactive")
            self.running = False
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.signals.log_message.emit(f"✅ Camera: {w}x{h}")

        # Set up MediaPipe
        self.signals.log_message.emit("🧠 Loading pose model...")
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        landmarker = PoseLandmarker.create_from_options(options)
        self.signals.log_message.emit("✅ Pose model loaded!")

        frame_timestamp_ms = 0
        check_interval = 0.15  # ~7 fps for smooth preview

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                frame_timestamp_ms += int(check_interval * 1000)
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # Draw landmarks on frame for preview
                display_frame = frame.copy()

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    display_frame = self._draw_landmarks(display_frame, landmarks)
                    self._process_landmarks(landmarks)
                else:
                    self._set_state("no_person")

                # Send frame to GUI
                self.signals.frame_ready.emit(display_frame)
                time.sleep(check_interval)
        except Exception as e:
            self.signals.log_message.emit(f"❌ Error: {e}")
        finally:
            landmarker.close()
            cap.release()
            self.signals.log_message.emit("🛑 Camera stopped.")

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        h, w = frame.shape[:2]

        nose = landmarks[NOSE]
        l_shoulder = landmarks[LEFT_SHOULDER]
        r_shoulder = landmarks[RIGHT_SHOULDER]

        # Draw key points
        points = [
            (nose, (0, 255, 200)),        # cyan-green for nose
            (l_shoulder, (255, 180, 0)),   # orange for shoulders
            (r_shoulder, (255, 180, 0)),
        ]

        for lm, color in points:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

        # Draw shoulder line
        if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5:
            p1 = (int(l_shoulder.x * w), int(l_shoulder.y * h))
            p2 = (int(r_shoulder.x * w), int(r_shoulder.y * h))
            cv2.line(frame, p1, p2, (255, 180, 0), 3)

            # Draw midpoint to nose line
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            nose_pt = (int(nose.x * w), int(nose.y * h))

            line_color = (0, 255, 100) if self.state == "good" else (0, 100, 255) if self.state == "slouching" else (200, 200, 200)
            cv2.line(frame, mid, nose_pt, line_color, 3)

        # Status overlay
        status_colors = {
            "good": (0, 200, 80),
            "slouching": (0, 80, 255),
            "calibrating": (0, 180, 255),
            "no_person": (128, 128, 128),
            "inactive": (128, 128, 128),
        }
        color = status_colors.get(self.state, (200, 200, 200))

        label = self.state.upper()
        if self.state == "slouching":
            label = f"SLOUCHING (dev: {self.current_deviation:.3f})"
        elif self.state == "calibrating":
            label = f"CALIBRATING ({len(self.calibration_samples)}/{self.calibration_count})"

        cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def _process_landmarks(self, landmarks):
        nose = landmarks[NOSE]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        if nose.visibility < 0.5 or left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            self._set_state("no_person")
            return

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
        nose_to_shoulder = shoulder_mid_y - nose.y

        if not self.calibrated:
            self.calibration_samples.append(nose_to_shoulder)
            self.signals.calibration_progress.emit(len(self.calibration_samples), self.calibration_count)

            if len(self.calibration_samples) >= self.calibration_count:
                self.baseline_ratio = sum(self.calibration_samples) / len(self.calibration_samples)
                self.calibrated = True
                self._set_state("good")
                self.signals.log_message.emit(f"✅ Calibrated! Baseline: {self.baseline_ratio:.4f}")
            return

        deviation = self.baseline_ratio - nose_to_shoulder
        self.current_deviation = deviation

        if deviation > self.prefs.sensitivity:
            if self.state != "slouching":
                self.slouch_start = time.time()
                self.signals.log_message.emit(f"⚠️ Slouching! (dev: {deviation:.3f})")
                self._trigger_alert()
            self._set_state("slouching", deviation)
        else:
            if self.state == "slouching" and self.slouch_start is not None:
                duration = time.time() - self.slouch_start
                self.logger.log(duration)
                self.slouch_start = None
                self.signals.log_message.emit(f"✅ Corrected after {duration:.1f}s")
            self._set_state("good", deviation)

    def _trigger_alert(self):
        now = time.time()
        if now - self.last_alert_time < self.prefs.cooldown_seconds:
            return
        self.last_alert_time = now

        if self.prefs.enable_sound:
            subprocess.run(["afplay", "/System/Library/Sounds/Funk.aiff"], capture_output=True)

        if self.prefs.enable_notification:
            script = 'display notification "You\'re slouching! Sit up straight 🪑" with title "PostureGuard" sound name "Funk"'
            subprocess.run(["osascript", "-e", script], capture_output=True)

    def _set_state(self, state: str, deviation: float = 0.0):
        self.state = state
        self.current_deviation = deviation
        self.signals.state_changed.emit(state, deviation)


# ─────────────────────────────────────────────
# Stylesheet
# ─────────────────────────────────────────────

STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    color: #e0e0e0;
    font-family: "SF Pro Display", "Helvetica Neue", sans-serif;
}
QTabWidget::pane {
    border: 1px solid #2a2a4a;
    background: #16163a;
    border-radius: 8px;
}
QTabBar::tab {
    background: #1a1a2e;
    color: #888;
    padding: 10px 20px;
    border: none;
    font-size: 13px;
    font-weight: 600;
}
QTabBar::tab:selected {
    color: #6cf;
    border-bottom: 2px solid #6cf;
}
QTabBar::tab:hover {
    color: #adf;
}
QPushButton {
    background-color: #2a2a5a;
    color: #e0e0e0;
    border: 1px solid #3a3a6a;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3a3a7a;
    border-color: #6cf;
}
QPushButton:pressed {
    background-color: #1a1a4a;
}
QPushButton#startBtn {
    background-color: #1a6a3a;
    border-color: #2a8a4a;
    font-size: 15px;
    padding: 12px 30px;
}
QPushButton#startBtn:hover {
    background-color: #2a8a4a;
}
QPushButton#stopBtn {
    background-color: #6a1a1a;
    border-color: #8a2a2a;
    font-size: 15px;
    padding: 12px 30px;
}
QPushButton#stopBtn:hover {
    background-color: #8a2a2a;
}
QProgressBar {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    background: #0e0e2a;
    height: 14px;
    text-align: center;
    color: #aaa;
    font-size: 10px;
}
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2a7a5a, stop:1 #6cf);
}
QCheckBox {
    font-size: 13px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #3a3a6a;
    background: #1a1a2e;
}
QCheckBox::indicator:checked {
    background: #6cf;
    border-color: #6cf;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #2a2a4a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background: #6cf;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #4a8abf;
    border-radius: 3px;
}
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #6cf;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLabel#statusLabel {
    font-size: 16px;
    font-weight: 700;
    padding: 8px;
}
QLabel#bigStat {
    font-size: 28px;
    font-weight: 700;
    color: #6cf;
}
QLabel#statLabel {
    font-size: 11px;
    color: #888;
}
QFrame#card {
    background: #1e1e42;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px;
}
QFrame#logFrame {
    background: #0e0e1e;
    border: 1px solid #1a1a3a;
    border-radius: 6px;
    padding: 8px;
}
"""


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.setWindowTitle("PostureGuard")
        self.setFixedSize(520, 680)
        self.setStyleSheet(STYLESHEET)

        self.prefs = Preferences.load()
        self.logger = PostureLogger()
        self.monitor = PostureMonitor(self.prefs, self.logger, model_path)

        # Connect signals
        self.monitor.signals.state_changed.connect(self._on_state_change)
        self.monitor.signals.frame_ready.connect(self._on_frame)
        self.monitor.signals.calibration_progress.connect(self._on_calibration)
        self.monitor.signals.log_message.connect(self._on_log)

        self._setup_ui()
        self._setup_tray()

        # Timer to refresh stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start(5000)

    def _setup_tray(self):
        """Set up system tray icon."""
        self.tray = QSystemTrayIcon(self)
        # Create a simple colored icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor("#6cf"))
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#1a1a2e"), 2))
        painter.setFont(QFont("Arial", 18))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🧍")
        painter.end()
        self.tray.setIcon(QIcon(pixmap))

        tray_menu = QMenu()
        show_action = QAction("Show PostureGuard", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)
        self.tray.setContextMenu(tray_menu)
        self.tray.activated.connect(self._tray_clicked)
        self.tray.show()

    def _tray_clicked(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show()
            self.raise_()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("PostureGuard")
        title.setFont(QFont("SF Pro Display", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #6cf;")
        header.addWidget(title)

        header.addStretch()

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #555; font-size: 18px;")
        header.addWidget(self.status_dot)

        self.status_label = QLabel("Not Monitoring")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet("color: #888;")
        header.addWidget(self.status_label)

        main_layout.addLayout(header)

        # ── Tabs ──
        tabs = QTabWidget()
        tabs.addTab(self._create_monitor_tab(), "📷  Monitor")
        tabs.addTab(self._create_stats_tab(), "📊  Stats")
        tabs.addTab(self._create_settings_tab(), "⚙️  Settings")
        main_layout.addWidget(tabs)

    # ── Monitor Tab ──

    def _create_monitor_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Camera preview
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(480, 270)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(
            "background: #0a0a1a; border: 2px solid #2a2a4a; border-radius: 10px;"
        )
        self.camera_label.setText("Camera preview will appear here")
        layout.addWidget(self.camera_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Calibration progress
        self.cal_bar = QProgressBar()
        self.cal_bar.setRange(0, 20)
        self.cal_bar.setValue(0)
        self.cal_bar.setFormat("Calibration: %v/%m")
        self.cal_bar.setVisible(False)
        layout.addWidget(self.cal_bar)

        # Deviation meter
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Posture deviation:"))
        self.dev_bar = QProgressBar()
        self.dev_bar.setRange(0, 100)
        self.dev_bar.setValue(0)
        self.dev_bar.setFormat("%v%")
        dev_layout.addWidget(self.dev_bar)
        self.dev_value_label = QLabel("0.000")
        self.dev_value_label.setStyleSheet("color: #6cf; font-family: monospace; font-size: 13px;")
        dev_layout.addWidget(self.dev_value_label)
        layout.addLayout(dev_layout)

        # Quick stats row
        stats_frame = QFrame()
        stats_frame.setObjectName("card")
        stats_row = QHBoxLayout(stats_frame)

        self.today_count_label = QLabel("0")
        self.today_count_label.setObjectName("bigStat")
        self.today_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_col = QVBoxLayout()
        count_col.addWidget(self.today_count_label)
        count_sub = QLabel("Slouches Today")
        count_sub.setObjectName("statLabel")
        count_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_col.addWidget(count_sub)
        stats_row.addLayout(count_col)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("color: #2a2a4a;")
        stats_row.addWidget(div)

        self.today_time_label = QLabel("0s")
        self.today_time_label.setObjectName("bigStat")
        self.today_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_col = QVBoxLayout()
        time_col.addWidget(self.today_time_label)
        time_sub = QLabel("Total Slouch Time")
        time_sub.setObjectName("statLabel")
        time_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_col.addWidget(time_sub)
        stats_row.addLayout(time_col)

        layout.addWidget(stats_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Monitoring")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_monitoring)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._stop_monitoring)
        self.stop_btn.setVisible(False)
        btn_layout.addWidget(self.stop_btn)

        self.recal_btn = QPushButton("🔄 Recalibrate")
        self.recal_btn.clicked.connect(self._recalibrate)
        self.recal_btn.setVisible(False)
        btn_layout.addWidget(self.recal_btn)

        layout.addLayout(btn_layout)

        # Log area
        log_frame = QFrame()
        log_frame.setObjectName("logFrame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(8, 4, 8, 4)
        self.log_label = QLabel("Ready. Click Start Monitoring to begin.")
        self.log_label.setWordWrap(True)
        self.log_label.setStyleSheet("font-size: 11px; color: #888; font-family: monospace;")
        self.log_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log_label.setMinimumHeight(48)
        log_layout.addWidget(self.log_label)
        layout.addWidget(log_frame)

        return tab

    # ── Stats Tab ──

    def _create_stats_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # Weekly chart
        week_group = QGroupBox("Last 7 Days")
        week_layout = QVBoxLayout(week_group)
        self.week_bars_widget = QWidget()
        self.week_bars_layout = QHBoxLayout(self.week_bars_widget)
        self.week_bars_layout.setSpacing(6)
        week_layout.addWidget(self.week_bars_widget)
        layout.addWidget(week_group)

        # Today's events
        events_group = QGroupBox("Today's Events")
        events_layout = QVBoxLayout(events_group)
        self.events_scroll = QScrollArea()
        self.events_scroll.setWidgetResizable(True)
        self.events_scroll.setStyleSheet("border: none; background: transparent;")
        self.events_container = QWidget()
        self.events_list_layout = QVBoxLayout(self.events_container)
        self.events_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.events_scroll.setWidget(self.events_container)
        events_layout.addWidget(self.events_scroll)
        layout.addWidget(events_group)

        clear_btn = QPushButton("🗑  Clear All Data")
        clear_btn.clicked.connect(self._clear_data)
        layout.addWidget(clear_btn)

        self._refresh_stats()
        return tab

    # ── Settings Tab ──

    def _create_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Notifications
        notif_group = QGroupBox("Alerts")
        notif_layout = QVBoxLayout(notif_group)

        self.notif_check = QCheckBox("macOS Notifications")
        self.notif_check.setChecked(self.prefs.enable_notification)
        self.notif_check.toggled.connect(lambda v: self._update_pref("enable_notification", v))
        notif_layout.addWidget(self.notif_check)

        self.sound_check = QCheckBox("Sound Alert (Funk)")
        self.sound_check.setChecked(self.prefs.enable_sound)
        self.sound_check.toggled.connect(lambda v: self._update_pref("enable_sound", v))
        notif_layout.addWidget(self.sound_check)

        test_btn = QPushButton("🔔 Test Alert")
        test_btn.clicked.connect(self._test_alert)
        notif_layout.addWidget(test_btn)

        layout.addWidget(notif_group)

        # Sensitivity
        sens_group = QGroupBox("Sensitivity")
        sens_layout = QVBoxLayout(sens_group)

        sens_header = QHBoxLayout()
        sens_header.addWidget(QLabel("Slouch threshold:"))
        self.sens_value_label = QLabel(self._sensitivity_label(self.prefs.sensitivity))
        self.sens_value_label.setStyleSheet("color: #6cf; font-weight: 600;")
        sens_header.addStretch()
        sens_header.addWidget(self.sens_value_label)
        sens_layout.addLayout(sens_header)

        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(20, 100)  # maps to 0.02–0.10
        self.sens_slider.setValue(int(self.prefs.sensitivity * 1000))
        self.sens_slider.valueChanged.connect(self._on_sensitivity_change)
        sens_layout.addWidget(self.sens_slider)

        sens_hint_layout = QHBoxLayout()
        hint_left = QLabel("More sensitive")
        hint_left.setStyleSheet("font-size: 10px; color: #666;")
        hint_right = QLabel("Less sensitive")
        hint_right.setStyleSheet("font-size: 10px; color: #666;")
        sens_hint_layout.addWidget(hint_left)
        sens_hint_layout.addStretch()
        sens_hint_layout.addWidget(hint_right)
        sens_layout.addLayout(sens_hint_layout)

        layout.addWidget(sens_group)

        # Cooldown
        cd_group = QGroupBox("Alert Cooldown")
        cd_layout = QVBoxLayout(cd_group)

        cd_header = QHBoxLayout()
        cd_header.addWidget(QLabel("Seconds between alerts:"))
        self.cd_value_label = QLabel(f"{self.prefs.cooldown_seconds}s")
        self.cd_value_label.setStyleSheet("color: #6cf; font-weight: 600;")
        cd_header.addStretch()
        cd_header.addWidget(self.cd_value_label)
        cd_layout.addLayout(cd_header)

        self.cd_slider = QSlider(Qt.Orientation.Horizontal)
        self.cd_slider.setRange(10, 120)
        self.cd_slider.setValue(self.prefs.cooldown_seconds)
        self.cd_slider.valueChanged.connect(self._on_cooldown_change)
        cd_layout.addWidget(self.cd_slider)

        layout.addWidget(cd_group)

        # Privacy note
        privacy = QLabel(
            "🔒 All processing happens on-device. No images or data are "
            "transmitted. Camera frames are analyzed in memory and discarded."
        )
        privacy.setWordWrap(True)
        privacy.setStyleSheet("font-size: 11px; color: #555; padding: 8px;")
        layout.addWidget(privacy)

        layout.addStretch()
        return tab

    # ── Actions ──

    def _start_monitoring(self):
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.recal_btn.setVisible(True)
        self.cal_bar.setVisible(True)
        self.cal_bar.setValue(0)
        self.monitor.start()

    def _stop_monitoring(self):
        self.monitor.stop()
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.recal_btn.setVisible(False)
        self.cal_bar.setVisible(False)
        self.camera_label.setText("Camera preview will appear here")
        self.camera_label.setPixmap(QPixmap())
        self._refresh_stats()

    def _recalibrate(self):
        self.cal_bar.setValue(0)
        self.cal_bar.setVisible(True)
        self.monitor.recalibrate()

    def _test_alert(self):
        if self.prefs.enable_sound:
            subprocess.run(["afplay", "/System/Library/Sounds/Funk.aiff"], capture_output=True)
        if self.prefs.enable_notification:
            script = 'display notification "This is a test alert!" with title "PostureGuard" sound name "Funk"'
            subprocess.run(["osascript", "-e", script], capture_output=True)

    def _clear_data(self):
        self.logger.clear()
        self._refresh_stats()

    # ── Signal Handlers ──

    def _on_state_change(self, state: str, deviation: float):
        state_display = {
            "inactive": ("Not Monitoring", "#555"),
            "calibrating": ("Calibrating — sit straight!", "#ffaa00"),
            "good": ("Good Posture ✓", "#4caf50"),
            "slouching": ("Slouching!", "#f44336"),
            "no_person": ("No Person Detected", "#888"),
        }
        text, color = state_display.get(state, ("Unknown", "#888"))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")

        # Update deviation bar
        dev_pct = min(int(abs(deviation) / 0.15 * 100), 100) if self.monitor.calibrated else 0
        self.dev_bar.setValue(dev_pct)
        self.dev_value_label.setText(f"{deviation:.3f}")

        if dev_pct < 40:
            self.dev_bar.setStyleSheet("QProgressBar::chunk { background: #4caf50; border-radius: 5px; }")
        elif dev_pct < 70:
            self.dev_bar.setStyleSheet("QProgressBar::chunk { background: #ff9800; border-radius: 5px; }")
        else:
            self.dev_bar.setStyleSheet("QProgressBar::chunk { background: #f44336; border-radius: 5px; }")

        # Update tray tooltip
        self.tray.setToolTip(f"PostureGuard — {text}")

        # Refresh stats periodically
        self._refresh_stats()

    def _on_frame(self, frame: np.ndarray):
        """Display camera frame in the preview label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def _on_calibration(self, current: int, total: int):
        self.cal_bar.setRange(0, total)
        self.cal_bar.setValue(current)
        if current >= total:
            self.cal_bar.setVisible(False)

    def _on_log(self, message: str):
        # Show last 3 lines
        current = self.log_label.text()
        lines = current.split("\n")
        lines.append(message)
        self.log_label.setText("\n".join(lines[-3:]))

    # ── Settings Handlers ──

    def _update_pref(self, key: str, value):
        setattr(self.prefs, key, value)
        self.monitor.prefs = self.prefs
        self.prefs.save()

    def _on_sensitivity_change(self, value: int):
        sens = value / 1000.0
        self.prefs.sensitivity = sens
        self.monitor.prefs.sensitivity = sens
        self.sens_value_label.setText(self._sensitivity_label(sens))
        self.prefs.save()

    def _on_cooldown_change(self, value: int):
        self.prefs.cooldown_seconds = value
        self.monitor.prefs.cooldown_seconds = value
        self.cd_value_label.setText(f"{value}s")
        self.prefs.save()

    @staticmethod
    def _sensitivity_label(val: float) -> str:
        if val < 0.03:
            return "Very High (0.{:.0f})".format(val * 100)
        if val < 0.05:
            return "High ({:.3f})".format(val)
        if val < 0.07:
            return "Medium ({:.3f})".format(val)
        return "Low ({:.3f})".format(val)

    # ── Stats Refresh ──

    def _refresh_stats(self):
        count, total = self.logger.today_stats()
        self.today_count_label.setText(str(count))
        self.today_time_label.setText(self._fmt(total))

        # Weekly bars
        while self.week_bars_layout.count():
            child = self.week_bars_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        weekly = self.logger.weekly_stats()
        max_total = max((s[2] for s in weekly), default=1) or 1

        for day_label, day_count, day_total in weekly:
            col = QVBoxLayout()
            col.setAlignment(Qt.AlignmentFlag.AlignBottom)

            # Value label
            val = QLabel(self._fmt(day_total) if day_total > 0 else "—")
            val.setStyleSheet("font-size: 9px; color: #888;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(val)

            # Bar
            bar_height = max(4, int(day_total / max_total * 80)) if day_total > 0 else 4
            if day_total == 0:
                bar_color = "#2a2a4a"
            elif day_total < 300:
                bar_color = "#4caf50"
            elif day_total < 900:
                bar_color = "#ff9800"
            else:
                bar_color = "#f44336"

            bar = QFrame()
            bar.setFixedSize(40, bar_height)
            bar.setStyleSheet(f"background: {bar_color}; border-radius: 4px;")
            col.addWidget(bar, alignment=Qt.AlignmentFlag.AlignCenter)

            # Day label
            day = QLabel(day_label)
            day.setStyleSheet("font-size: 11px; color: #aaa;")
            day.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(day)

            wrapper = QWidget()
            wrapper.setLayout(col)
            self.week_bars_layout.addWidget(wrapper)

        # Today's events list
        while self.events_list_layout.count():
            child = self.events_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        events = self.logger.today_events()
        if not events:
            empty = QLabel("No slouching events today ✨")
            empty.setStyleSheet("color: #555; font-size: 12px; padding: 16px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.events_list_layout.addWidget(empty)
        else:
            for event in reversed(events[-15:]):
                t = datetime.fromisoformat(event.timestamp).strftime("%H:%M")
                row = QLabel(f"  ⚠️  {t}    →    {event.formatted_duration}")
                row.setStyleSheet(
                    "font-size: 12px; color: #ccc; padding: 4px 8px; "
                    "background: #1a1a3a; border-radius: 4px; margin: 2px 0;"
                )
                self.events_list_layout.addWidget(row)

    @staticmethod
    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    # ── Window Events ──

    def closeEvent(self, event):
        """Minimize to tray instead of quitting."""
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "PostureGuard",
            "Running in the background. Click the tray icon to reopen.",
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Dependency check
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        missing.append("PyQt6")

    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        print(f"Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)

    print("🧍 PostureGuard starting...\n")

    # Request camera on main thread
    print("📷 Requesting camera access...")
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret:
            print(f"✅ Camera OK ({frame.shape[1]}x{frame.shape[0]})")
        test_cap.release()
    else:
        test_cap.release()
        print("❌ Camera access denied!")
        print(f"   Grant camera access to: {sys.executable}")
        print("   System Settings → Privacy & Security → Camera")
        sys.exit(1)

    # Download model
    model_path = ensure_model()

    # Launch GUI
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # keep running in tray

    window = MainWindow(model_path)
    window.show()

    sys.exit(app.exec())
