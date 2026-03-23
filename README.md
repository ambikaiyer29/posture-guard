# PostureGuard v2 🧍

A macOS desktop app that monitors your posture in real-time using your webcam and alerts you when you're slouching, leaning forward, tilting your head, or sitting with uneven shoulders. Built with Python, MediaPipe, and PyQt6.

All processing happens 100% on-device. No images or data are ever transmitted.


## Quick Start

```bash
# 1. Navigate to the project folder
cd PostureGuardPython

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python3 postureguard.py
```

On first launch, the app downloads a ~4MB pose detection model and requests camera access. Grant permission when macOS prompts you.


## Building as a macOS App (Recommended)

To run PostureGuard as a proper macOS application (no terminal needed, runs in background):

```bash
chmod +x build_app.sh
./build_app.sh
```

This creates `PostureGuard.app`. You can then:

```bash
# Launch it
open PostureGuard.app

# Install permanently
mv PostureGuard.app /Applications/

# Start at login
# System Settings → General → Login Items → click + → select PostureGuard
```

When launched as an `.app`, camera permission shows "PostureGuard" by name (not "python3"), and all logs go to `~/.postureguard/app.log`.


## How It Works

The app uses MediaPipe's PoseLandmarker to track 13 body landmarks (nose, eyes, ears, shoulders, elbows, wrists, hips) via your webcam. It calculates three posture metrics by comparing your current pose against a calibrated baseline:

**Forward Lean** — measures the vertical distance between your nose and the midpoint of your shoulders. When you slouch or lean forward, your head drops and this distance shrinks.

**Head Tilt** — measures the vertical difference between your left and right ears. When you tilt your head to one side, this value increases.

**Shoulder Asymmetry** — measures the vertical difference between your left and right shoulders. When one shoulder drops or you lean to one side, this value increases.

The detection system includes several layers to prevent false positives from brief movements like scratching your nose or glancing sideways:

- **Rolling average** smooths values over the last 8 frames (~1 second), filtering out momentary spikes.
- **4-second confirmation window** requires bad posture to persist continuously before triggering an alert. Quick movements are ignored.
- **Grace period** prevents a slouch event from ending prematurely — you need 5 consecutive good frames to confirm you've actually corrected your posture.


## Features

### 📷 Monitor Tab

- **Live camera preview** with full skeleton overlay drawn on your body in real-time (head, shoulders, arms, hips connected with lines).
- Skeleton turns **green** for good posture and **red/orange** when issues are detected, with different shades per posture type.
- **Calibration progress bar** — sit up straight for ~3 seconds while the app learns your baseline.
- **Deviation meter** showing how far you are from your calibrated posture (green → orange → red).
- **Today's stats** — slouch count and total slouch time displayed as large cards.
- **Daily posture score** — percentage of monitoring time spent with good posture, shown in the header.
- **Focus mode** button — pauses all alerts while still tracking posture (useful during meetings or presentations).
- **Log area** showing real-time events (calibration, detections, corrections).

### 📊 Stats Tab

- **7-day bar chart** — color-coded weekly breakdown (green < 5 min slouching, orange < 15 min, red > 15 min).
- **Hourly heatmap** — a 24-hour color strip showing when you slouch most during the day, helping you identify problem times.
- **Slouch type breakdown** — shows how much time was spent on each posture issue (forward lean, head tilt, uneven shoulders).
- **Event log** — scrollable list of today's slouch events with timestamps, durations, and posture types.
- **Export CSV** — saves all data (up to 30 days) as a CSV file with timestamp, duration, and posture type columns.
- **Clear All Data** — wipes all events and resets achievements.

### 🏆 Badges Tab

Eight achievement badges to earn through consistent good posture:

| Badge | Name | How to Unlock |
|-------|------|---------------|
| 🥉 | First Hour | 1 hour of good posture in a day |
| 🥈 | Iron Spine | 2 hours of good posture in a day |
| 🥇 | Posture Master | 4 hours of good posture in a day |
| 🔥 | Hat Trick | 3-day streak with <10 min slouching |
| ⚡ | Week Warrior | 7-day streak with <10 min slouching |
| 👑 | Legendary | 30-day streak with <10 min slouching |
| ✨ | Perfect Day | Zero slouch events in a full monitoring day |
| 🎯 | Focused Hour | 1 hour of monitoring with zero slouching |

A **day streak counter** tracks consecutive days with under 10 minutes of slouching. Only days where the app was actually used count — days with no monitoring don't inflate the streak.

A macOS notification pops up when you unlock a new badge.

### ⚙️ Settings Tab

**Alerts**
- Toggle macOS notifications on/off.
- Toggle sound alerts on/off (plays the system "Funk" sound).
- **Test Alert** button to preview what notifications look and sound like.

**Posture Detection**
- Toggle skeleton overlay on camera preview.
- Toggle head tilt detection on/off.
- Toggle shoulder asymmetry detection on/off.

**Sensitivity**
- Slouch threshold slider (0.020–0.100). Lower values trigger more easily, higher values are more forgiving. Default: 0.070.

**Alert Cooldown**
- Minimum seconds between repeated alerts (10s–120s). Default: 30s.

**Break Reminders**
- Toggle stand-up/stretch reminders on/off.
- Configurable interval (5–120 minutes). Default: 30 minutes.
- Sends a macOS notification reminding you to take a break.

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+M` | Toggle monitoring on/off |
| `Ctrl+Shift+F` | Toggle focus mode (mute/unmute alerts) |
| `Ctrl+Shift+R` | Recalibrate baseline posture |

### 🔔 Posture-Specific Alerts

Each posture issue has its own notification message:

- **Forward Lean** → "You're leaning forward! Sit back and straighten up 🪑"
- **Head Tilt** → "Your head is tilting! Level it out 🧠"
- **Uneven Shoulders** → "Your shoulders are uneven! Balance them out 💪"

### 📌 Background Mode & System Tray

- **Closing the window** minimizes the app to the system tray — monitoring continues in the background.
- **Click the tray icon** (🧍) to reopen the window.
- **Right-click the tray icon** for Show/Quit options.
- The app only fully quits when you choose "Quit" from the menu or tray.


## File Structure

```
PostureGuardPython/
├── postureguard.py       # The entire app (single file)
├── requirements.txt      # Python dependencies
├── build_app.sh          # Builds a macOS .app bundle
├── setup.py              # Alternative: py2app packaging
└── README.md             # This file
```

All user data is stored in `~/.postureguard/`:

```
~/.postureguard/
├── pose_landmarker_lite.task   # ML model (downloaded on first run)
├── prefs.json                  # User preferences
├── posture_log.json            # Slouch events (30-day retention)
├── achievements.json           # Unlocked badges
└── app.log                     # Application log
```


## Requirements

- macOS 13.0 (Ventura) or later
- Python 3.9+
- Webcam (built-in FaceTime camera or external)

### Python Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Webcam capture and frame processing |
| mediapipe | Pose landmark detection (ML model) |
| PyQt6 | GUI framework (window, tray, widgets) |
| numpy | Array operations for frame data |


## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera access denied | System Settings → Privacy & Security → Camera → enable your terminal app or PostureGuard.app |
| "No Person Detected" | Ensure your face and shoulders are visible to the camera. Check lighting. |
| Too many false alerts | Settings → increase sensitivity slider toward "Less sensitive", or increase cooldown |
| Not detecting bad posture | Settings → decrease sensitivity slider toward "More sensitive" |
| Head tilt alerts too sensitive | Settings → uncheck "Detect head tilt", or increase threshold in prefs.json |
| No notifications | System Settings → Notifications → allow for PostureGuard or Script Editor |
| App won't start as .app | Check `~/.postureguard/app.log` for errors |
| Achievements unlocked incorrectly | Click "Clear Data" in Stats tab to reset everything |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your activated venv |
| Camera permission shows "python3" not "PostureGuard" | Use `build_app.sh` to create a proper .app bundle |


## Tips for Best Results

- **Calibrate in your normal working position** — sit how you normally would when you have good posture. Don't sit unnaturally straight.
- **Recalibrate after moving** — if you adjust your chair, desk, or laptop position, hit Recalibrate (or Ctrl+Shift+R).
- **Good lighting matters** — the pose model works best with even lighting on your face and shoulders. Avoid strong backlighting.
- **Camera angle** — position the camera roughly at eye level for the most accurate readings. Laptop cameras looking up from below can skew shoulder measurements.
- **Use Focus Mode during meetings** — press Ctrl+Shift+F before a video call to suppress alerts while still tracking your posture.
- **Check the heatmap** — after a few days, the hourly heatmap reveals patterns (e.g., you always slouch after lunch). Use this to set targeted break reminders.


## Privacy

- All pose detection runs **locally on your Mac** using Apple's GPU acceleration via MediaPipe.
- **No camera frames are saved** — each frame is analyzed in memory and immediately discarded.
- **No data is transmitted** — everything stays in `~/.postureguard/` on your machine.
- The only network request is the one-time model download from Google's public storage on first run.
