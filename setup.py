"""
setup.py — Package PostureGuard as a macOS .app bundle
Run: python setup.py py2app
"""

from setuptools import setup

APP = ['postureguard.py']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,  # will use default; replace with 'icon.icns' if you have one
    'plist': {
        'CFBundleName': 'PostureGuard',
        'CFBundleDisplayName': 'PostureGuard',
        'CFBundleIdentifier': 'com.postureguard.app',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0',
        'LSMinimumSystemVersion': '13.0',
        'NSCameraUsageDescription': 'PostureGuard uses the camera to monitor your posture. No images are stored or transmitted.',
        'NSHumanReadableCopyright': 'PostureGuard © 2026',
        'LSUIElement': False,  # False = shows in Dock; True = menu bar only
    },
    'packages': ['cv2', 'mediapipe', 'PyQt6', 'numpy'],
    'includes': [
        'mediapipe.tasks',
        'mediapipe.tasks.vision',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
