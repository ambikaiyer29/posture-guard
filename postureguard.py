"""
PostureGuard v2 — Enhanced macOS Posture Monitor
Features: live skeleton overlay, multiple posture types, break reminders,
daily score, heatmap, CSV export, achievements, focus mode, keyboard shortcuts.
"""

from __future__ import annotations

import csv
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import json
import math
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QTabWidget,
    QFrame, QProgressBar, QScrollArea, QSystemTrayIcon, QMenu,
    QGroupBox, QFileDialog, QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QFont, QColor, QPainter, QPen,
    QAction, QKeySequence, QShortcut, QBrush, QLinearGradient
)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILENAME = "pose_landmarker_lite.task"
DATA_DIR = os.path.expanduser("~/.postureguard")

# MediaPipe landmark indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
]

# Achievement definitions
ACHIEVEMENTS = [
    {"id": "first_hour", "name": "First Hour", "desc": "1 hour of good posture in a day", "icon": "🥉", "threshold_seconds": 3600},
    {"id": "two_hours", "name": "Iron Spine", "desc": "2 hours of good posture in a day", "icon": "🥈", "threshold_seconds": 7200},
    {"id": "four_hours", "name": "Posture Master", "desc": "4 hours of good posture in a day", "icon": "🥇", "threshold_seconds": 14400},
    {"id": "streak_3", "name": "Hat Trick", "desc": "3-day streak with <10 min slouching", "icon": "🔥", "threshold_days": 3},
    {"id": "streak_7", "name": "Week Warrior", "desc": "7-day streak with <10 min slouching", "icon": "⚡", "threshold_days": 7},
    {"id": "streak_30", "name": "Legendary", "desc": "30-day streak with <10 min slouching", "icon": "👑", "threshold_days": 30},
    {"id": "perfect_day", "name": "Perfect Day", "desc": "Zero slouch events in a full day", "icon": "✨", "threshold_zero": True},
    {"id": "no_slouch_1h", "name": "Focused Hour", "desc": "1 hour monitoring with zero slouching", "icon": "🎯", "threshold_clean_hour": True},
]


def ensure_model() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, MODEL_FILENAME)
    if not os.path.exists(path):
        print("📥 Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("✅ Model downloaded.")
    return path


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class SlouchEvent:
    timestamp: str
    duration_seconds: float
    posture_type: str = "general"  # general, forward_lean, head_tilt, shoulder_asymmetry

    @property
    def formatted_duration(self) -> str:
        m, s = divmod(int(self.duration_seconds), 60)
        h, m = divmod(m, 60)
        if h: return f"{h}h {m}m"
        if m: return f"{m}m {s}s"
        return f"{s}s"


@dataclass
class Preferences:
    enable_notification: bool = True
    enable_sound: bool = True
    sensitivity: float = 0.07
    cooldown_seconds: int = 30
    break_interval_minutes: int = 30
    break_reminders_enabled: bool = True
    focus_mode: bool = False
    show_skeleton: bool = True
    detect_head_tilt: bool = True
    detect_forward_lean: bool = True
    detect_shoulder_asymmetry: bool = True
    head_tilt_threshold: float = 0.08
    shoulder_asym_threshold: float = 0.05

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, "prefs.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> Preferences:
        try:
            with open(os.path.join(DATA_DIR, "prefs.json")) as f:
                data = json.load(f)
                # Handle missing keys from older prefs files
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()


# ─────────────────────────────────────────────
# Posture Logger + Achievements + Heatmap
# ─────────────────────────────────────────────

class PostureLogger:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.log_path = os.path.join(DATA_DIR, "posture_log.json")
        self.achievements_path = os.path.join(DATA_DIR, "achievements.json")
        self.events: list[SlouchEvent] = self._load()
        self.unlocked_achievements: list[str] = self._load_achievements()
        self.monitoring_start: Optional[float] = None
        self.total_monitoring_seconds: float = 0
        self.good_posture_seconds: float = 0
        self._last_good_check: float = 0

    def start_session(self):
        self.monitoring_start = time.time()
        self._last_good_check = time.time()

    def tick_good_posture(self):
        now = time.time()
        if self._last_good_check > 0:
            self.good_posture_seconds += now - self._last_good_check
        self._last_good_check = now

    def tick_bad_posture(self):
        self._last_good_check = time.time()

    def stop_session(self):
        if self.monitoring_start:
            self.total_monitoring_seconds += time.time() - self.monitoring_start
            self.monitoring_start = None

    def log(self, duration: float, posture_type: str = "general"):
        if duration < 3.0:
            return
        self.events.append(SlouchEvent(
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
            posture_type=posture_type,
        ))
        self._save()
        self._check_achievements()

    def daily_score(self) -> float:
        """Returns percentage of good posture time (0-100)."""
        total = self.total_monitoring_seconds
        if total < 60:
            return 100.0
        good = self.good_posture_seconds
        return min(100.0, max(0.0, (good / total) * 100))

    def today_stats(self) -> tuple[int, float]:
        today = datetime.now().date()
        te = [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == today]
        return len(te), sum(e.duration_seconds for e in te)

    def weekly_stats(self) -> list[tuple[str, int, float]]:
        results = []
        for i in range(6, -1, -1):
            day = datetime.now().date() - timedelta(days=i)
            de = [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == day]
            results.append((day.strftime("%a"), len(de), sum(e.duration_seconds for e in de)))
        return results

    def hourly_heatmap(self) -> list[float]:
        """Returns 24 values (one per hour) of total slouch seconds for today."""
        today = datetime.now().date()
        hours = [0.0] * 24
        for e in self.events:
            dt = datetime.fromisoformat(e.timestamp)
            if dt.date() == today:
                hours[dt.hour] += e.duration_seconds
        return hours

    def posture_type_breakdown(self) -> dict[str, float]:
        """Today's slouching broken down by type."""
        today = datetime.now().date()
        breakdown: dict[str, float] = {}
        for e in self.events:
            if datetime.fromisoformat(e.timestamp).date() == today:
                t = e.posture_type or "general"
                breakdown[t] = breakdown.get(t, 0) + e.duration_seconds
        return breakdown

    def today_events(self) -> list[SlouchEvent]:
        today = datetime.now().date()
        return [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == today]

    def clear(self):
        self.events.clear()
        self.unlocked_achievements.clear()
        self.good_posture_seconds = 0
        self.total_monitoring_seconds = 0
        self._save()
        self._save_achievements()

    def export_csv(self, filepath: str):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Duration (seconds)", "Posture Type"])
            for e in self.events:
                writer.writerow([e.timestamp, e.duration_seconds, e.posture_type])

    def streak_days(self) -> int:
        """Count consecutive days with <10 min slouching, only counting days the app was used."""
        # First, find all days that have any logged events (proof the app was running)
        active_days = set()
        for e in self.events:
            active_days.add(datetime.fromisoformat(e.timestamp).date())
        # Also count today if we're currently monitoring (even with 0 slouches)
        if self.monitoring_start is not None:
            active_days.add(datetime.now().date())

        streak = 0
        for i in range(0, 60):
            day = datetime.now().date() - timedelta(days=i)
            if day not in active_days:
                # Day with no app usage — skip today (might still be building data)
                # but break the streak for past days
                if i == 0:
                    continue  # give today a pass, user might have just started
                break

            de = [e for e in self.events if datetime.fromisoformat(e.timestamp).date() == day]
            total = sum(e.duration_seconds for e in de)
            if total < 600:  # < 10 min
                streak += 1
            else:
                break
        return streak

    def _check_achievements(self):
        newly_unlocked = []
        for ach in ACHIEVEMENTS:
            if ach["id"] in self.unlocked_achievements:
                continue

            unlocked = False
            if "threshold_seconds" in ach:
                if self.good_posture_seconds >= ach["threshold_seconds"]:
                    unlocked = True
            elif "threshold_days" in ach:
                if self.streak_days() >= ach["threshold_days"]:
                    unlocked = True
            elif "threshold_zero" in ach:
                count, _ = self.today_stats()
                if count == 0 and self.total_monitoring_seconds > 3600:
                    unlocked = True
            elif "threshold_clean_hour" in ach:
                if self.good_posture_seconds >= 3600:
                    count, _ = self.today_stats()
                    if count == 0:
                        unlocked = True

            if unlocked:
                self.unlocked_achievements.append(ach["id"])
                newly_unlocked.append(ach)

        if newly_unlocked:
            self._save_achievements()
            for ach in newly_unlocked:
                subprocess.run(["osascript", "-e",
                    f'display notification "Achievement: {ach["name"]} — {ach["desc"]}" '
                    f'with title "PostureGuard {ach["icon"]}" sound name "Glass"'
                ], capture_output=True)

    def _save(self):
        cutoff = datetime.now() - timedelta(days=30)
        self.events = [e for e in self.events if datetime.fromisoformat(e.timestamp) > cutoff]
        with open(self.log_path, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)

    def _load(self) -> list[SlouchEvent]:
        try:
            with open(self.log_path) as f:
                data = json.load(f)
                events = []
                for d in data:
                    if "posture_type" not in d:
                        d["posture_type"] = "general"
                    events.append(SlouchEvent(**d))
                return events
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_achievements(self):
        with open(self.achievements_path, "w") as f:
            json.dump(self.unlocked_achievements, f)

    def _load_achievements(self) -> list[str]:
        try:
            with open(self.achievements_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []


# ─────────────────────────────────────────────
# Signal Bridge
# ─────────────────────────────────────────────

class MonitorSignals(QObject):
    state_changed = pyqtSignal(str, float, str)   # state, deviation, posture_type
    frame_ready = pyqtSignal(np.ndarray)
    calibration_progress = pyqtSignal(int, int)
    log_message = pyqtSignal(str)


# ─────────────────────────────────────────────
# Posture Monitor — Multi-type Detection
# ─────────────────────────────────────────────

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
        self.current_posture_type = ""

        self.calibration_samples: list[dict] = []
        self.baseline: dict = {}
        self.calibration_count = 20

        self.slouch_start: Optional[float] = None
        self.slouch_type: str = "general"
        self.last_alert_time: float = 0
        self._thread: Optional[threading.Thread] = None

        # ── Smoothing & grace period ──
        # Rolling buffer of recent metrics (smooths out noise from scratching, glancing, etc.)
        self._metrics_buffer: list[dict] = []
        self._buffer_size = 8  # average over ~1 second of frames at 0.12s interval

        # Confirmation window: bad posture must persist for N seconds before triggering
        self._bad_posture_since: Optional[float] = None  # when bad posture first appeared
        self._confirmation_seconds = 4.0  # must be bad for this long before it counts

        # Grace period: brief good frames during a slouch don't reset the slouch
        self._good_frames_in_slouch = 0
        self._grace_frames = 5  # need this many consecutive good frames to end a slouch

    def start(self):
        if self.running:
            return
        self.running = True
        self.calibrated = False
        self.calibration_samples.clear()
        self._set_state("calibrating")
        self.logger.start_session()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self.slouch_start is not None:
            self.logger.log(time.time() - self.slouch_start, self.slouch_type)
            self.slouch_start = None
        self.logger.stop_session()
        self._set_state("inactive")

    def recalibrate(self):
        self.calibrated = False
        self.calibration_samples.clear()
        self._metrics_buffer.clear()
        self._bad_posture_since = None
        self._good_frames_in_slouch = 0
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
        self.signals.log_message.emit("✅ Ready!")

        frame_ts = 0
        interval = 0.12

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_ts += int(interval * 1000)
                result = landmarker.detect_for_video(mp_image, frame_ts)

                display = frame.copy()

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms = result.pose_landmarks[0]
                    if self.prefs.show_skeleton:
                        display = self._draw_skeleton(display, lms)
                    self._analyze_posture(lms)
                else:
                    self._set_state("no_person")

                self.signals.frame_ready.emit(display)
                time.sleep(interval)
        except Exception as e:
            self.signals.log_message.emit(f"❌ {e}")
        finally:
            landmarker.close()
            cap.release()

    def _draw_skeleton(self, frame: np.ndarray, landmarks) -> np.ndarray:
        h, w = frame.shape[:2]

        # Draw connections
        for i1, i2 in SKELETON_CONNECTIONS:
            lm1, lm2 = landmarks[i1], landmarks[i2]
            if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                p1 = (int(lm1.x * w), int(lm1.y * h))
                p2 = (int(lm2.x * w), int(lm2.y * h))

                if self.state == "slouching":
                    # Color by posture type
                    type_colors = {
                        "Forward Lean":     (0, 80, 255),   # red
                        "Head Tilt":        (0, 140, 255),  # orange
                        "Uneven Shoulders": (0, 100, 230),  # deep orange
                    }
                    color = type_colors.get(self.current_posture_type, (0, 80, 255))
                elif self.state == "good":
                    color = (0, 230, 120)  # green
                else:
                    color = (200, 200, 200)  # gray
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        # Draw joints
        key_joints = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
                      LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                      LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]
        for idx in key_joints:
            lm = landmarks[idx]
            if lm.visibility > 0.4:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, (100, 200, 255), 1, cv2.LINE_AA)

        # Status overlay
        if self.state == "slouching":
            type_labels = {
                "Forward Lean":      "FORWARD LEAN",
                "Head Tilt":         "HEAD TILT",
                "Uneven Shoulders":  "UNEVEN SHOULDERS",
            }
            label = type_labels.get(self.current_posture_type, "BAD POSTURE")
            color = (0, 80, 255)
        elif self.state == "calibrating":
            label = f"CALIBRATING ({len(self.calibration_samples)}/{self.calibration_count})"
            color = (0, 180, 255)
        elif self.state == "good":
            label = "GOOD POSTURE"
            color = (0, 200, 80)
        else:
            label = self.state.upper()
            color = (180, 180, 180)

        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return frame

    def _extract_metrics(self, lms) -> Optional[dict]:
        """Extract posture metrics from landmarks."""
        nose, l_sh, r_sh = lms[NOSE], lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
        l_ear, r_ear = lms[LEFT_EAR], lms[RIGHT_EAR]
        l_hip, r_hip = lms[LEFT_HIP], lms[RIGHT_HIP]

        if any(lm.visibility < 0.4 for lm in [nose, l_sh, r_sh]):
            return None

        sh_mid_y = (l_sh.y + r_sh.y) / 2.0
        sh_mid_x = (l_sh.x + r_sh.x) / 2.0

        return {
            "nose_to_shoulder_y": sh_mid_y - nose.y,
            "head_tilt": abs(l_ear.y - r_ear.y) if l_ear.visibility > 0.4 and r_ear.visibility > 0.4 else 0,
            "shoulder_asymmetry": abs(l_sh.y - r_sh.y),
            "nose_x_offset": abs(nose.x - sh_mid_x),
        }

    def _analyze_posture(self, lms):
        metrics = self._extract_metrics(lms)
        if metrics is None:
            self._set_state("no_person")
            return

        if not self.calibrated:
            self.calibration_samples.append(metrics)
            self.signals.calibration_progress.emit(len(self.calibration_samples), self.calibration_count)

            if len(self.calibration_samples) >= self.calibration_count:
                self.baseline = {}
                for key in metrics:
                    vals = [s[key] for s in self.calibration_samples]
                    self.baseline[key] = sum(vals) / len(vals)
                self.calibrated = True
                self._set_state("good")
                self.signals.log_message.emit("✅ Calibrated!")
            return

        # ── Step 1: Add to rolling buffer and compute smoothed metrics ──
        self._metrics_buffer.append(metrics)
        if len(self._metrics_buffer) > self._buffer_size:
            self._metrics_buffer.pop(0)

        # Average over the buffer to smooth out momentary movements
        smoothed = {}
        for key in metrics:
            smoothed[key] = sum(m[key] for m in self._metrics_buffer) / len(self._metrics_buffer)

        # ── Step 2: Check for posture issues using smoothed values ──
        issues = []

        # Forward lean (head dropping toward shoulders)
        dev = self.baseline["nose_to_shoulder_y"] - smoothed["nose_to_shoulder_y"]
        if dev > self.prefs.sensitivity:
            issues.append(("Forward Lean", dev))

        # Head tilt
        if self.prefs.detect_head_tilt:
            tilt_dev = smoothed["head_tilt"] - self.baseline["head_tilt"]
            if tilt_dev > self.prefs.head_tilt_threshold:
                issues.append(("Head Tilt", tilt_dev))

        # Shoulder asymmetry
        if self.prefs.detect_shoulder_asymmetry:
            asym_dev = smoothed["shoulder_asymmetry"] - self.baseline["shoulder_asymmetry"]
            if asym_dev > self.prefs.shoulder_asym_threshold:
                issues.append(("Uneven Shoulders", asym_dev))

        max_dev = max((d for _, d in issues), default=0)
        self.current_deviation = max_dev

        # ── Step 3: Apply confirmation window + grace period ──
        if issues:
            worst_type = max(issues, key=lambda x: x[1])[0]
            self._good_frames_in_slouch = 0  # reset grace counter

            if self.state == "slouching":
                # Already in slouch state — keep updating the type if it changed
                self.logger.tick_bad_posture()
                self._set_state("slouching", max_dev, worst_type)

            elif self._bad_posture_since is None:
                # First bad frame — start the confirmation timer
                self._bad_posture_since = time.time()
                # Don't change state yet — still in "good" visually
                self._set_state("good", max_dev)

            elif time.time() - self._bad_posture_since >= self._confirmation_seconds:
                # Bad posture confirmed — it's been sustained long enough
                self.slouch_start = time.time()
                self.slouch_type = worst_type.lower().replace(" ", "_")
                self.signals.log_message.emit(f"⚠️ {worst_type} detected (sustained {self._confirmation_seconds:.0f}s)")
                if not self.prefs.focus_mode:
                    self._trigger_alert()
                self.logger.tick_bad_posture()
                self._set_state("slouching", max_dev, worst_type)
            else:
                # Still in confirmation window — not yet triggered
                self._set_state("good", max_dev)

        else:
            # No issues detected on this frame
            self._bad_posture_since = None  # reset confirmation timer

            if self.state == "slouching":
                # Currently in slouch — apply grace period before ending it
                self._good_frames_in_slouch += 1

                if self._good_frames_in_slouch >= self._grace_frames:
                    # Sustained good posture — end the slouch
                    if self.slouch_start is not None:
                        dur = time.time() - self.slouch_start
                        type_label = self.slouch_type.replace("_", " ").title()
                        self.logger.log(dur, self.slouch_type)
                        self.slouch_start = None
                        self.signals.log_message.emit(f"✅ {type_label} corrected after {dur:.1f}s")
                    self._good_frames_in_slouch = 0
                    self.logger.tick_good_posture()
                    self._set_state("good", max_dev)
                else:
                    # Still in grace period — stay in slouch state
                    self.logger.tick_bad_posture()
                    self._set_state("slouching", max_dev, self.current_posture_type)
            else:
                self.logger.tick_good_posture()
                self._set_state("good", max_dev)

    def _trigger_alert(self):
        now = time.time()
        if now - self.last_alert_time < self.prefs.cooldown_seconds:
            return
        self.last_alert_time = now

        # Posture-type-specific messages
        alert_messages = {
            "forward_lean":      "You're leaning forward! Sit back and straighten up 🪑",
            "head_tilt":         "Your head is tilting! Level it out 🧠",
            "uneven_shoulders":  "Your shoulders are uneven! Balance them out 💪",
        }
        msg = alert_messages.get(self.slouch_type, "Check your posture! 🪑")

        if self.prefs.enable_sound:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Funk.aiff"])

        if self.prefs.enable_notification:
            script = f'display notification "{msg}" with title "PostureGuard" sound name "Funk"'
            subprocess.Popen(["osascript", "-e", script])

    def _set_state(self, state: str, deviation: float = 0.0, posture_type: str = ""):
        self.state = state
        self.current_deviation = deviation
        self.current_posture_type = posture_type
        self.signals.state_changed.emit(state, deviation, posture_type)


# ─────────────────────────────────────────────
# Stylesheet
# ─────────────────────────────────────────────

STYLE = """
QMainWindow { background: #12122a; }
QWidget { color: #ddd; font-family: "SF Pro Display", "Helvetica Neue", sans-serif; }
QTabWidget::pane { border: 1px solid #252550; background: #14143a; border-radius: 8px; }
QTabBar::tab { background: #12122a; color: #777; padding: 9px 16px; border: none; font-size: 12px; font-weight: 600; }
QTabBar::tab:selected { color: #6cf; border-bottom: 2px solid #6cf; }
QTabBar::tab:hover { color: #ade; }
QPushButton { background: #222258; color: #ddd; border: 1px solid #333370; border-radius: 7px; padding: 9px 18px; font-size: 12px; font-weight: 600; }
QPushButton:hover { background: #333380; border-color: #6cf; }
QPushButton#startBtn { background: #1a6a3a; border-color: #2a8a4a; font-size: 14px; padding: 11px 28px; }
QPushButton#startBtn:hover { background: #2a8a4a; }
QPushButton#stopBtn { background: #6a1a1a; border-color: #8a2a2a; font-size: 14px; padding: 11px 28px; }
QPushButton#stopBtn:hover { background: #8a2a2a; }
QPushButton#focusBtn { background: #5a3a8a; border-color: #7a5aaa; }
QPushButton#focusBtn:hover { background: #7a5aaa; }
QPushButton#focusBtnActive { background: #8a5a2a; border-color: #aa7a3a; }
QProgressBar { border: 1px solid #252550; border-radius: 5px; background: #0c0c22; height: 12px; text-align: center; color: #999; font-size: 9px; }
QProgressBar::chunk { border-radius: 4px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2a7a5a,stop:1 #6cf); }
QCheckBox { font-size: 12px; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 2px solid #333370; background: #12122a; }
QCheckBox::indicator:checked { background: #6cf; border-color: #6cf; }
QSlider::groove:horizontal { height: 5px; background: #252550; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #6cf; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #4a8abf; border-radius: 2px; }
QGroupBox { border: 1px solid #252550; border-radius: 7px; margin-top: 10px; padding-top: 18px; font-size: 12px; font-weight: 600; color: #6cf; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QFrame#card { background: #1a1a42; border: 1px solid #252550; border-radius: 8px; padding: 10px; }
QFrame#logFrame { background: #0a0a1a; border: 1px solid #181840; border-radius: 5px; padding: 6px; }
QScrollArea { border: none; background: transparent; }
QSpinBox { background: #1a1a3a; border: 1px solid #333370; border-radius: 4px; padding: 4px; color: #ddd; }
"""


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.setWindowTitle("PostureGuard")
        self.setFixedSize(560, 740)
        self.setStyleSheet(STYLE)

        self.prefs = Preferences.load()
        self.logger = PostureLogger()
        self.monitor = PostureMonitor(self.prefs, self.logger, model_path)

        self.monitor.signals.state_changed.connect(self._on_state_change)
        self.monitor.signals.frame_ready.connect(self._on_frame)
        self.monitor.signals.calibration_progress.connect(self._on_calibration)
        self.monitor.signals.log_message.connect(self._on_log)

        self._setup_ui()
        self._setup_tray()
        self._setup_shortcuts()
        self._setup_break_timer()

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start(5000)

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        px = QPixmap(32, 32)
        px.fill(QColor(0, 0, 0, 0))
        p = QPainter(px)
        p.setFont(QFont("Arial", 20))
        p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, "🧍")
        p.end()
        self.tray.setIcon(QIcon(px))
        menu = QMenu()
        menu.addAction("Show", self.show)
        menu.addAction("Quit", QApplication.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self.show() if r == QSystemTrayIcon.ActivationReason.Trigger else None)
        self.tray.show()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Shift+M"), self, self._toggle_monitoring)
        QShortcut(QKeySequence("Ctrl+Shift+F"), self, self._toggle_focus_mode)
        QShortcut(QKeySequence("Ctrl+Shift+R"), self, lambda: self.monitor.recalibrate() if self.monitor.running else None)

    def _setup_break_timer(self):
        self.break_timer = QTimer()
        self.break_timer.timeout.connect(self._break_reminder)
        self._update_break_timer()

    def _update_break_timer(self):
        if self.prefs.break_reminders_enabled and self.monitor.running:
            self.break_timer.start(self.prefs.break_interval_minutes * 60 * 1000)
        else:
            self.break_timer.stop()

    def _break_reminder(self):
        if self.prefs.focus_mode:
            return
        subprocess.Popen(["osascript", "-e",
            'display notification "Time to stand up and stretch! 🧘" with title "PostureGuard — Break Time" sound name "Glass"'])

    def _toggle_monitoring(self):
        if self.monitor.running:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _toggle_focus_mode(self):
        self.prefs.focus_mode = not self.prefs.focus_mode
        self.monitor.prefs.focus_mode = self.prefs.focus_mode
        self.prefs.save()
        self._update_focus_btn()
        self._on_log(f"{'🔕' if self.prefs.focus_mode else '🔔'} Focus mode {'ON' if self.prefs.focus_mode else 'OFF'}")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QVBoxLayout(central)
        ml.setContentsMargins(10, 10, 10, 10)
        ml.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("PostureGuard")
        title.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #6cf;")
        hdr.addWidget(title)
        hdr.addStretch()

        self.score_label = QLabel("Score: —")
        self.score_label.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: 700;")
        hdr.addWidget(self.score_label)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #555; font-size: 16px;")
        hdr.addWidget(self.status_dot)
        self.status_label = QLabel("Not Monitoring")
        self.status_label.setStyleSheet("color: #888; font-size: 13px; font-weight: 600;")
        hdr.addWidget(self.status_label)
        ml.addLayout(hdr)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._tab_monitor(), "📷 Monitor")
        tabs.addTab(self._tab_stats(), "📊 Stats")
        tabs.addTab(self._tab_achievements(), "🏆 Badges")
        tabs.addTab(self._tab_settings(), "⚙️ Settings")
        ml.addWidget(tabs)
        self.tabs = tabs

    # ── Monitor Tab ──

    def _tab_monitor(self) -> QWidget:
        tab = QWidget()
        ly = QVBoxLayout(tab)
        ly.setSpacing(8)

        self.cam_label = QLabel("Camera preview appears here")
        self.cam_label.setFixedSize(520, 292)
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setStyleSheet("background: #08081a; border: 2px solid #252550; border-radius: 8px;")
        ly.addWidget(self.cam_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.cal_bar = QProgressBar()
        self.cal_bar.setRange(0, 20)
        self.cal_bar.setFormat("Calibrating: %v/%m — sit up straight!")
        self.cal_bar.setVisible(False)
        ly.addWidget(self.cal_bar)

        # Deviation + posture type
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Deviation:"))
        self.dev_bar = QProgressBar()
        self.dev_bar.setRange(0, 100)
        self.dev_bar.setValue(0)
        dev_row.addWidget(self.dev_bar)
        self.posture_type_label = QLabel("")
        self.posture_type_label.setStyleSheet("color: #ff9800; font-size: 11px; font-weight: 600; min-width: 110px;")
        dev_row.addWidget(self.posture_type_label)
        ly.addLayout(dev_row)

        # Stats cards
        cards = QFrame()
        cards.setObjectName("card")
        cr = QHBoxLayout(cards)
        self.today_count_lbl = QLabel("0")
        self.today_count_lbl.setStyleSheet("font-size: 26px; font-weight: 700; color: #6cf;")
        self.today_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.today_time_lbl = QLabel("0s")
        self.today_time_lbl.setStyleSheet("font-size: 26px; font-weight: 700; color: #6cf;")
        self.today_time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for lbl, sub in [(self.today_count_lbl, "Slouches"), (self.today_time_lbl, "Total Time")]:
            col = QVBoxLayout()
            col.addWidget(lbl)
            s = QLabel(sub)
            s.setStyleSheet("font-size: 10px; color: #777;")
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(s)
            cr.addLayout(col)

        ly.addWidget(cards)

        # Buttons row
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Monitoring")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_monitoring)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._stop_monitoring)
        self.stop_btn.setVisible(False)
        btn_row.addWidget(self.stop_btn)

        self.recal_btn = QPushButton("🔄 Recalibrate")
        self.recal_btn.clicked.connect(lambda: self.monitor.recalibrate() if self.monitor.running else None)
        self.recal_btn.setVisible(False)
        btn_row.addWidget(self.recal_btn)

        self.focus_btn = QPushButton("🔕 Focus Mode")
        self.focus_btn.setObjectName("focusBtn")
        self.focus_btn.clicked.connect(self._toggle_focus_mode)
        self.focus_btn.setVisible(False)
        btn_row.addWidget(self.focus_btn)

        ly.addLayout(btn_row)

        # Shortcut hints
        hints = QLabel("Shortcuts:  Ctrl+Shift+M = toggle monitoring  |  Ctrl+Shift+F = focus mode  |  Ctrl+Shift+R = recalibrate")
        hints.setStyleSheet("font-size: 9px; color: #555; padding: 2px;")
        hints.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(hints)

        # Log
        lf = QFrame()
        lf.setObjectName("logFrame")
        ll = QVBoxLayout(lf)
        ll.setContentsMargins(6, 3, 6, 3)
        self.log_label = QLabel("Ready. Press Start or Ctrl+Shift+M.")
        self.log_label.setWordWrap(True)
        self.log_label.setStyleSheet("font-size: 10px; color: #777; font-family: monospace;")
        self.log_label.setMinimumHeight(36)
        ll.addWidget(self.log_label)
        ly.addWidget(lf)

        return tab

    # ── Stats Tab ──

    def _tab_stats(self) -> QWidget:
        tab = QWidget()
        ly = QVBoxLayout(tab)
        ly.setSpacing(10)

        # Weekly bars
        wg = QGroupBox("Last 7 Days")
        wl = QVBoxLayout(wg)
        self.week_widget = QWidget()
        self.week_layout = QHBoxLayout(self.week_widget)
        self.week_layout.setSpacing(4)
        wl.addWidget(self.week_widget)
        ly.addWidget(wg)

        # Heatmap
        hg = QGroupBox("Today's Hourly Heatmap")
        hl = QVBoxLayout(hg)
        self.heatmap_widget = QWidget()
        self.heatmap_widget.setFixedHeight(60)
        hl.addWidget(self.heatmap_widget)
        self.heatmap_widget.paintEvent = self._paint_heatmap
        ly.addWidget(hg)

        # Posture type breakdown
        bg = QGroupBox("Slouch Types Today")
        bl = QVBoxLayout(bg)
        self.breakdown_label = QLabel("No data yet")
        self.breakdown_label.setStyleSheet("font-size: 12px; color: #aaa; padding: 4px;")
        bl.addWidget(self.breakdown_label)
        ly.addWidget(bg)

        # Events
        eg = QGroupBox("Recent Events")
        el = QVBoxLayout(eg)
        self.events_scroll = QScrollArea()
        self.events_scroll.setWidgetResizable(True)
        self.events_container = QWidget()
        self.events_layout = QVBoxLayout(self.events_container)
        self.events_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.events_scroll.setWidget(self.events_container)
        self.events_scroll.setMaximumHeight(120)
        el.addWidget(self.events_scroll)
        ly.addWidget(eg)

        # Export + clear
        br = QHBoxLayout()
        exp_btn = QPushButton("📄 Export CSV")
        exp_btn.clicked.connect(self._export_csv)
        br.addWidget(exp_btn)
        clr_btn = QPushButton("🗑 Clear Data")
        clr_btn.clicked.connect(lambda: (self.logger.clear(), self._refresh_stats()))
        br.addWidget(clr_btn)
        ly.addLayout(br)

        self._refresh_stats()
        return tab

    # ── Achievements Tab ──

    def _tab_achievements(self) -> QWidget:
        tab = QWidget()
        ly = QVBoxLayout(tab)
        ly.setSpacing(8)

        # Streak
        streak_frame = QFrame()
        streak_frame.setObjectName("card")
        sl = QHBoxLayout(streak_frame)
        self.streak_label = QLabel("0")
        self.streak_label.setStyleSheet("font-size: 36px; font-weight: 700; color: #ff9800;")
        sl.addWidget(self.streak_label)
        sl2 = QVBoxLayout()
        sl2.addWidget(QLabel("Day Streak"))
        streak_sub = QLabel("Consecutive days with <10 min slouching")
        streak_sub.setStyleSheet("font-size: 10px; color: #777;")
        sl2.addWidget(streak_sub)
        sl.addLayout(sl2)
        sl.addStretch()
        ly.addWidget(streak_frame)

        # Badges grid
        self.badges_scroll = QScrollArea()
        self.badges_scroll.setWidgetResizable(True)
        self.badges_container = QWidget()
        self.badges_layout = QVBoxLayout(self.badges_container)
        self.badges_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.badges_scroll.setWidget(self.badges_container)
        ly.addWidget(self.badges_scroll)

        self._refresh_achievements()
        return tab

    # ── Settings Tab ──

    def _tab_settings(self) -> QWidget:
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        ly = QVBoxLayout(inner)
        ly.setSpacing(12)

        # Alerts
        ag = QGroupBox("Alerts")
        al = QVBoxLayout(ag)
        self.notif_chk = QCheckBox("macOS Notifications")
        self.notif_chk.setChecked(self.prefs.enable_notification)
        self.notif_chk.toggled.connect(lambda v: self._set_pref("enable_notification", v))
        al.addWidget(self.notif_chk)
        self.sound_chk = QCheckBox("Sound Alert")
        self.sound_chk.setChecked(self.prefs.enable_sound)
        self.sound_chk.toggled.connect(lambda v: self._set_pref("enable_sound", v))
        al.addWidget(self.sound_chk)
        test_btn = QPushButton("🔔 Test Alert")
        test_btn.clicked.connect(lambda: subprocess.Popen(["afplay", "/System/Library/Sounds/Funk.aiff"]))
        al.addWidget(test_btn)
        ly.addWidget(ag)

        # Detection types
        dg = QGroupBox("Posture Detection")
        dl = QVBoxLayout(dg)
        self.skel_chk = QCheckBox("Show skeleton overlay on camera")
        self.skel_chk.setChecked(self.prefs.show_skeleton)
        self.skel_chk.toggled.connect(lambda v: self._set_pref("show_skeleton", v))
        dl.addWidget(self.skel_chk)
        self.tilt_chk = QCheckBox("Detect head tilt")
        self.tilt_chk.setChecked(self.prefs.detect_head_tilt)
        self.tilt_chk.toggled.connect(lambda v: self._set_pref("detect_head_tilt", v))
        dl.addWidget(self.tilt_chk)
        self.asym_chk = QCheckBox("Detect shoulder asymmetry")
        self.asym_chk.setChecked(self.prefs.detect_shoulder_asymmetry)
        self.asym_chk.toggled.connect(lambda v: self._set_pref("detect_shoulder_asymmetry", v))
        dl.addWidget(self.asym_chk)
        ly.addWidget(dg)

        # Sensitivity
        sg = QGroupBox("Sensitivity")
        sgl = QVBoxLayout(sg)
        sh = QHBoxLayout()
        sh.addWidget(QLabel("Slouch threshold:"))
        self.sens_lbl = QLabel(f"{self.prefs.sensitivity:.3f}")
        self.sens_lbl.setStyleSheet("color: #6cf; font-weight: 600;")
        sh.addStretch()
        sh.addWidget(self.sens_lbl)
        sgl.addLayout(sh)
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(20, 100)
        self.sens_slider.setValue(int(self.prefs.sensitivity * 1000))
        self.sens_slider.valueChanged.connect(self._on_sens)
        sgl.addWidget(self.sens_slider)
        hh = QHBoxLayout()
        l1 = QLabel("More sensitive")
        l1.setStyleSheet("font-size: 9px; color: #555;")
        l2 = QLabel("Less sensitive")
        l2.setStyleSheet("font-size: 9px; color: #555;")
        hh.addWidget(l1); hh.addStretch(); hh.addWidget(l2)
        sgl.addLayout(hh)
        ly.addWidget(sg)

        # Cooldown
        cg = QGroupBox("Alert Cooldown")
        cl = QVBoxLayout(cg)
        crr = QHBoxLayout()
        crr.addWidget(QLabel("Seconds between alerts:"))
        self.cd_lbl = QLabel(f"{self.prefs.cooldown_seconds}s")
        self.cd_lbl.setStyleSheet("color: #6cf; font-weight: 600;")
        crr.addStretch()
        crr.addWidget(self.cd_lbl)
        cl.addLayout(crr)
        self.cd_slider = QSlider(Qt.Orientation.Horizontal)
        self.cd_slider.setRange(10, 120)
        self.cd_slider.setValue(self.prefs.cooldown_seconds)
        self.cd_slider.valueChanged.connect(self._on_cd)
        cl.addWidget(self.cd_slider)
        ly.addWidget(cg)

        # Break reminders
        brg = QGroupBox("Break Reminders")
        brl = QVBoxLayout(brg)
        self.break_chk = QCheckBox("Remind me to stand up & stretch")
        self.break_chk.setChecked(self.prefs.break_reminders_enabled)
        self.break_chk.toggled.connect(lambda v: (self._set_pref("break_reminders_enabled", v), self._update_break_timer()))
        brl.addWidget(self.break_chk)
        bir = QHBoxLayout()
        bir.addWidget(QLabel("Interval (minutes):"))
        self.break_spin = QSpinBox()
        self.break_spin.setRange(5, 120)
        self.break_spin.setValue(self.prefs.break_interval_minutes)
        self.break_spin.valueChanged.connect(lambda v: (self._set_pref("break_interval_minutes", v), self._update_break_timer()))
        bir.addWidget(self.break_spin)
        brl.addLayout(bir)
        ly.addWidget(brg)

        # Privacy
        p = QLabel("🔒 All processing on-device. No data transmitted. Camera frames discarded after analysis.")
        p.setWordWrap(True)
        p.setStyleSheet("font-size: 10px; color: #444; padding: 6px;")
        ly.addWidget(p)
        ly.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return tab

    # ── Actions ──

    def _start_monitoring(self):
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.recal_btn.setVisible(True)
        self.focus_btn.setVisible(True)
        self.cal_bar.setVisible(True)
        self.cal_bar.setValue(0)
        self.monitor.start()
        self._update_break_timer()

    def _stop_monitoring(self):
        self.monitor.stop()
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.recal_btn.setVisible(False)
        self.focus_btn.setVisible(False)
        self.cal_bar.setVisible(False)
        self.cam_label.setPixmap(QPixmap())
        self.cam_label.setText("Camera preview appears here")
        self.break_timer.stop()
        self._refresh_stats()

    def _update_focus_btn(self):
        if self.prefs.focus_mode:
            self.focus_btn.setText("🔔 Alerts On")
            self.focus_btn.setObjectName("focusBtnActive")
        else:
            self.focus_btn.setText("🔕 Focus Mode")
            self.focus_btn.setObjectName("focusBtn")
        self.focus_btn.setStyleSheet(self.focus_btn.styleSheet())  # force refresh

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "postureguard_export.csv", "CSV (*.csv)")
        if path:
            self.logger.export_csv(path)
            self._on_log(f"📄 Exported to {path}")

    # ── Signal Handlers ──

    def _on_state_change(self, state: str, deviation: float, posture_type: str):
        display = {
            "inactive":    ("Not Monitoring",  "#555"),
            "calibrating": ("Calibrating...",   "#ffaa00"),
            "good":        ("Good Posture ✓",   "#4caf50"),
            "no_person":   ("No Person",        "#888"),
        }

        if state == "slouching" and posture_type:
            # Use the specific posture issue as the status text
            type_display = {
                "Forward Lean":      ("Forward Lean ↘",       "#f44336"),
                "Head Tilt":         ("Head Tilting ↗",       "#ff6600"),
                "Uneven Shoulders":  ("Uneven Shoulders ⤵",  "#e65100"),
            }
            text, color = type_display.get(posture_type, ("Bad Posture", "#f44336"))
        elif state == "slouching":
            text, color = "Bad Posture", "#f44336"
        else:
            text, color = display.get(state, ("Unknown", "#888"))

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600;")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 16px;")

        self.posture_type_label.setText(posture_type if state == "slouching" else "")

        pct = min(int(abs(deviation) / 0.15 * 100), 100) if self.monitor.calibrated else 0
        self.dev_bar.setValue(pct)
        chunk_color = "#4caf50" if pct < 40 else "#ff9800" if pct < 70 else "#f44336"
        self.dev_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {chunk_color}; border-radius: 4px; }}")

        score = self.logger.daily_score()
        sc = "#4caf50" if score > 80 else "#ff9800" if score > 50 else "#f44336"
        self.score_label.setText(f"Score: {score:.0f}%")
        self.score_label.setStyleSheet(f"color: {sc}; font-size: 14px; font-weight: 700;")

    def _on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(img).scaled(self.cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.cam_label.setPixmap(px)

    def _on_calibration(self, cur: int, total: int):
        self.cal_bar.setRange(0, total)
        self.cal_bar.setValue(cur)
        if cur >= total:
            self.cal_bar.setVisible(False)

    def _on_log(self, msg: str):
        lines = self.log_label.text().split("\n")
        lines.append(msg)
        self.log_label.setText("\n".join(lines[-3:]))

    # ── Settings Helpers ──

    def _set_pref(self, key: str, val):
        setattr(self.prefs, key, val)
        setattr(self.monitor.prefs, key, val)
        self.prefs.save()

    def _on_sens(self, v):
        s = v / 1000.0
        self._set_pref("sensitivity", s)
        self.sens_lbl.setText(f"{s:.3f}")

    def _on_cd(self, v):
        self._set_pref("cooldown_seconds", v)
        self.cd_lbl.setText(f"{v}s")

    # ── Stats Refresh ──

    def _refresh_stats(self):
        count, total = self.logger.today_stats()
        self.today_count_lbl.setText(str(count))
        self.today_time_lbl.setText(self._fmt(total))

        # Weekly bars
        while self.week_layout.count():
            c = self.week_layout.takeAt(0)
            if c.widget(): c.widget().deleteLater()

        weekly = self.logger.weekly_stats()
        mx = max((s[2] for s in weekly), default=1) or 1
        for dl, dc, dt in weekly:
            col = QVBoxLayout()
            col.setAlignment(Qt.AlignmentFlag.AlignBottom)
            v = QLabel(self._fmt(dt) if dt else "—")
            v.setStyleSheet("font-size: 8px; color: #777;")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(v)
            bh = max(4, int(dt / mx * 70)) if dt else 4
            bc = "#2a2a4a" if not dt else "#4caf50" if dt < 300 else "#ff9800" if dt < 900 else "#f44336"
            bar = QFrame()
            bar.setFixedSize(35, bh)
            bar.setStyleSheet(f"background: {bc}; border-radius: 3px;")
            col.addWidget(bar, alignment=Qt.AlignmentFlag.AlignCenter)
            d = QLabel(dl)
            d.setStyleSheet("font-size: 10px; color: #999;")
            d.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(d)
            w = QWidget()
            w.setLayout(col)
            self.week_layout.addWidget(w)

        # Breakdown
        bd = self.logger.posture_type_breakdown()
        if bd:
            parts = [f"{k.replace('_', ' ').title()}: {self._fmt(v)}" for k, v in bd.items()]
            self.breakdown_label.setText("  •  ".join(parts))
        else:
            self.breakdown_label.setText("No data yet")

        # Events list
        while self.events_layout.count():
            c = self.events_layout.takeAt(0)
            if c.widget(): c.widget().deleteLater()

        evts = self.logger.today_events()
        if not evts:
            el = QLabel("No events today ✨")
            el.setStyleSheet("color: #555; font-size: 11px; padding: 10px;")
            el.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.events_layout.addWidget(el)
        else:
            for e in reversed(evts[-12:]):
                t = datetime.fromisoformat(e.timestamp).strftime("%H:%M")
                tp = e.posture_type.replace("_", " ").title() if e.posture_type != "general" else ""
                txt = f"⚠️ {t}  →  {e.formatted_duration}"
                if tp:
                    txt += f"  ({tp})"
                r = QLabel(txt)
                r.setStyleSheet("font-size: 11px; color: #bbb; background: #16163a; border-radius: 3px; padding: 3px 6px; margin: 1px 0;")
                self.events_layout.addWidget(r)

        # Heatmap repaint
        self.heatmap_widget.update()

        # Achievements
        self._refresh_achievements()

    def _refresh_achievements(self):
        if not hasattr(self, 'badges_layout'):
            return
        while self.badges_layout.count():
            c = self.badges_layout.takeAt(0)
            if c.widget(): c.widget().deleteLater()

        self.streak_label.setText(str(self.logger.streak_days()))

        for ach in ACHIEVEMENTS:
            unlocked = ach["id"] in self.logger.unlocked_achievements
            frame = QFrame()
            frame.setObjectName("card")
            row = QHBoxLayout(frame)

            icon = QLabel(ach["icon"])
            icon.setStyleSheet(f"font-size: 24px; {'opacity: 1' if unlocked else 'opacity: 0.3'};")
            row.addWidget(icon)

            info = QVBoxLayout()
            name = QLabel(ach["name"])
            name.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {'#6cf' if unlocked else '#555'};")
            info.addWidget(name)
            desc = QLabel(ach["desc"])
            desc.setStyleSheet(f"font-size: 10px; color: {'#aaa' if unlocked else '#444'};")
            info.addWidget(desc)
            row.addLayout(info)
            row.addStretch()

            status = QLabel("✅ Unlocked" if unlocked else "🔒 Locked")
            status.setStyleSheet(f"font-size: 10px; color: {'#4caf50' if unlocked else '#555'};")
            row.addWidget(status)

            self.badges_layout.addWidget(frame)

    # ── Heatmap Painting ──

    def _paint_heatmap(self, event):
        w = self.heatmap_widget
        painter = QPainter(w)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        hours = self.logger.hourly_heatmap()
        mx = max(hours) if max(hours) > 0 else 1
        cell_w = (w.width() - 20) / 24
        cell_h = 28
        y = 8

        for i, val in enumerate(hours):
            x = 10 + i * cell_w
            intensity = val / mx if mx > 0 else 0

            if val == 0:
                color = QColor(30, 30, 60)
            else:
                r = int(50 + intensity * 200)
                g = int(200 - intensity * 150)
                b = int(80 - intensity * 60)
                color = QColor(min(r, 255), max(g, 0), max(b, 0))

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(20, 20, 50), 1))
            painter.drawRoundedRect(int(x), y, int(cell_w - 1), cell_h, 3, 3)

            # Hour labels (every 3 hours)
            if i % 3 == 0:
                painter.setPen(QColor(120, 120, 120))
                painter.setFont(QFont("Helvetica", 8))
                painter.drawText(int(x), y + cell_h + 14, f"{i:02d}")

        painter.end()

    # ── Utilities ──

    @staticmethod
    def _fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h: return f"{h}h {m}m"
        if m: return f"{m}m {sec}s"
        return f"{sec}s"

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray.showMessage("PostureGuard", "Running in background. Click tray to reopen.",
                              QSystemTrayIcon.MessageIcon.Information, 2000)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import logging

    # Set up logging to file (useful when running as .app without a terminal)
    log_dir = os.path.expanduser("~/.postureguard")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger("PostureGuard")

    missing = []
    for mod in ["cv2", "mediapipe", "PyQt6"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod if mod != "cv2" else "opencv-python")
    if missing:
        log.error(f"Missing: {', '.join(missing)}")
        # If PyQt6 is available, show a dialog; otherwise just print
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "PostureGuard", f"Missing dependencies: {', '.join(missing)}\n\nRun: pip install {' '.join(missing)}")
        except Exception:
            pass
        sys.exit(1)

    log.info("PostureGuard v2 starting...")

    # Camera auth on main thread (required by macOS)
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"
    cap = cv2.VideoCapture(0)
    camera_ok = False
    if cap.isOpened():
        ret, f = cap.read()
        if ret:
            log.info(f"Camera OK ({f.shape[1]}x{f.shape[0]})")
            camera_ok = True
        cap.release()
    else:
        cap.release()

    if not camera_ok:
        log.error("Camera access denied")
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication(sys.argv)
            QMessageBox.warning(None, "PostureGuard — Camera Access",
                f"Camera access was denied.\n\n"
                f"Please grant camera permission:\n"
                f"  System Settings → Privacy & Security → Camera\n\n"
                f"Look for: {sys.executable}\n\n"
                f"Then relaunch PostureGuard.")
        except Exception:
            pass
        sys.exit(1)

    model_path = ensure_model()

    app = QApplication(sys.argv)
    app.setApplicationName("PostureGuard")
    app.setQuitOnLastWindowClosed(False)  # keep running in tray when window is closed

    win = MainWindow(model_path)
    win.show()

    log.info("App ready.")
    sys.exit(app.exec())
