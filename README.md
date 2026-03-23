# PostureGuard 🧍

A lightweight macOS menu bar app that monitors your posture via webcam and alerts you when you're slouching. Built with Python, OpenCV, and MediaPipe.

## Quick Start

```bash
# 1. Clone or download this folder

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run it
python postureguard.py
```

That's it. A **🧍** icon appears in your menu bar.

## First Use

1. Click the menu bar icon → **▶️ Start Monitoring**
2. **Grant camera access** when macOS prompts you (System Settings → Privacy → Camera → Terminal/iTerm)
3. **Sit up straight** — the app calibrates for ~3 seconds
4. Status changes to **✅ Good Posture** — you're live!

## How It Works

```
Webcam frame → MediaPipe Pose → Extract nose + shoulders
    → Compare to calibrated baseline → Deviation too high?
        → Yes: Mark as slouching, trigger alert, start timer
        → No:  Good posture, log any previous slouch duration
```

The key metric is the **vertical distance between your nose and the midpoint of your shoulders**. When you slouch, your head drops forward/down, reducing this distance relative to your calibrated baseline.

## Menu Bar Features

| Menu Item | What it does |
|-----------|-------------|
| **Status line** | Shows current state (Good / Slouching / Calibrating / No Person) |
| **Start/Stop** | Toggle monitoring on/off |
| **Recalibrate** | Re-learn your "good posture" baseline |
| **📊 Today** | Slouch count and total slouch time for today |
| **📅 This Week** | 7-day breakdown with mini bar chart |
| **🕐 Recent Events** | Last 10 slouch events with time and duration |
| **⚙️ Settings** | Notifications, sound, sensitivity, cooldown |
| **🗑 Clear All Data** | Wipe all logged events |

## Settings

### Notification Methods
- **Notifications**: macOS banner notifications (uses `osascript`)
- **Sound**: Plays the system "Funk" sound

### Sensitivity
Controls how much deviation from baseline triggers a slouch alert:
- **Very High** (0.025) — triggers easily, good if you want strict monitoring
- **High** (0.04) — default, catches most slouching
- **Medium** (0.06) — more forgiving
- **Low** (0.09) — only triggers on significant slouching

### Alert Cooldown
Minimum time between repeated alerts: 10s / 30s / 1min / 2min

## Data Storage

All data is stored locally in `~/.postureguard/`:
- `prefs.json` — your settings
- `posture_log.json` — slouch events (timestamp + duration)
- Auto-cleans events older than 30 days

**No images or video are ever saved.** Frames are processed in memory and discarded.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera won't open | Grant camera access: System Settings → Privacy & Security → Camera → enable your terminal app |
| "No Person Detected" | Make sure your face + shoulders are visible to the webcam. Check lighting. |
| Too many false alerts | Go to Settings → Sensitivity → set to Medium or Low |
| Not catching slouches | Settings → Sensitivity → Very High |
| No notifications | System Settings → Notifications → Script Editor → allow |
| `rumps` install fails | `pip install pyobjc-framework-Cocoa` first, then retry |

## Running at Login (Optional)

To auto-start PostureGuard when you log in:

1. Create a shell script:
```bash
echo '#!/bin/bash
cd ~/path/to/PostureGuardPython
source venv/bin/activate
python postureguard.py' > ~/postureguard_start.sh
chmod +x ~/postureguard_start.sh
```

2. Add it to Login Items:
   - System Settings → General → Login Items → add `postureguard_start.sh`

Or use `launchd` for a more robust setup.

## Architecture

```
postureguard.py
├── SlouchEvent          — data class for a single slouch event
├── Preferences          — settings with JSON persistence
├── PostureLogger        — event logging, daily/weekly stats, 30-day retention
├── PostureMonitor       — webcam capture + MediaPipe pose analysis (runs on background thread)
└── PostureGuardApp      — rumps menu bar app with full UI
```

Everything is in one file for simplicity. Feel free to split it up as the project grows.
