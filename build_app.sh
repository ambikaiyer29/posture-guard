#!/bin/bash
# build_app.sh — Creates a macOS .app bundle for PostureGuard
# This wraps the Python script in a proper .app so it:
#   - Has its own camera permission entry
#   - Can be added to Login Items
#   - Runs without a visible terminal window
#   - Lives in /Applications or your Desktop

set -e

APP_NAME="PostureGuard"
APP_DIR="$APP_NAME.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔨 Building $APP_NAME.app..."

# Clean previous build
rm -rf "$APP_DIR"

# Create .app bundle structure
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# ── Info.plist ──
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>PostureGuard</string>
    <key>CFBundleDisplayName</key>
    <string>PostureGuard</string>
    <key>CFBundleIdentifier</key>
    <string>com.postureguard.app</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>PostureGuard</string>
    <key>CFBundleIconFile</key>
    <string>icon</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSCameraUsageDescription</key>
    <string>PostureGuard uses the camera to monitor your posture and alert you when slouching. No images are stored or transmitted.</string>
    <key>NSHumanReadableCopyright</key>
    <string>PostureGuard © 2026</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.healthcare-fitness</string>
</dict>
</plist>
PLIST

# ── Launcher script ──
# This is the executable that macOS runs when you open the .app
# It finds the Python venv and launches postureguard.py

cat > "$APP_DIR/Contents/MacOS/PostureGuard" << LAUNCHER
#!/bin/bash
# PostureGuard launcher — finds the venv and runs the app

SCRIPT_DIR="$SCRIPT_DIR"
LOG_FILE="\$HOME/.postureguard/app.log"
mkdir -p "\$HOME/.postureguard"

# Activate venv if present
if [ -f "\$SCRIPT_DIR/venv/bin/activate" ]; then
    source "\$SCRIPT_DIR/venv/bin/activate"
fi

# Run the app, logging output for debugging
cd "\$SCRIPT_DIR"
exec python3 "\$SCRIPT_DIR/postureguard.py" >> "\$LOG_FILE" 2>&1
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/PostureGuard"

# ── Create a simple icon (optional) ──
# If you have an icon.icns file, copy it:
# cp icon.icns "$APP_DIR/Contents/Resources/icon.icns"

echo ""
echo "✅ Built: $APP_DIR"
echo ""
echo "To use it:"
echo "  1. Double-click $APP_DIR to launch (or: open $APP_DIR)"
echo "  2. macOS will ask for camera permission — click Allow"
echo "  3. The app window opens + a tray icon appears"
echo "  4. Close the window → app keeps running in background via tray icon"
echo ""
echo "To install permanently:"
echo "  mv $APP_DIR /Applications/"
echo ""
echo "To start at login:"
echo "  System Settings → General → Login Items → add PostureGuard"
echo ""
