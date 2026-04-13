#!/bin/bash
# WOODSHED — Setup script for Raspberry Pi 5
# Run once: bash setup.sh

set -e
echo "=== WOODSHED setup ==="

# System deps
sudo apt-get update -q
sudo apt-get install -y \
    python3-pip \
    python3-pygame \
    rubberband-cli \
    libsndfile1 \
    portaudio19-dev \
    libatlas-base-dev \
    fonts-dejavu-core

# Python deps
pip3 install --break-system-packages \
    pyrubberband \
    soundfile \
    numpy \
    librosa \
    scipy \
    sounddevice

echo ""
echo "=== Done! ==="
echo "Run the app:  python3 woodshed.py"
echo "With a file:  python3 woodshed.py ~/Music/mysong.mp3"
echo ""
echo "Keyboard shortcuts:"
echo "  SPACE       Play / Pause"
echo "  LEFT/RIGHT  Seek ±5 seconds"
echo "  ESC         Quit"
