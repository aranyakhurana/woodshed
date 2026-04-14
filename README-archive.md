WOODSHED 🎸
Guitar Practice Tool — Phase 1

Raspberry Pi 5 + 7" touchscreen
What it does
Feature 	Notes
Speed 	25–200% independent of pitch (Rubber Band library)
Pitch 	±12 semitones independent of speed
A/B Loop 	Tap [A and B] buttons to set region, toggle LOOP
Markers 	Named markers, tap to jump, saved with session
EQ 	Low / Mid / High shelf (±12dB each)
Speed Trainer 	Auto-ramps tempo step by step, N reps per step
Session save/load 	Saves everything to ~/woodshed_session.json
File browser 	Browse and open MP3, WAV, FLAC, AIFF, OGG, M4A
Setup

git clone <your-repo> woodshed
cd woodshed
bash setup.sh

Running

# Basic
python3 woodshed.py

# With a file preloaded
python3 woodshed.py ~/Music/BB_King_TheThrill.mp3

Autostart on Pi boot (optional)

Add to /etc/rc.local before exit 0:

DISPLAY=:0 python3 /home/pi/woodshed/woodshed.py &

Or create a systemd service for cleaner handling.
Audio output → HX Stomp

    Connect Behringer UCA202 (or similar USB interface) to Pi USB
    Set as default output: sudo raspi-config → Audio → USB Audio
    Run a stereo cable from UCA202 L/R out to HX Stomp FX Return (L+R)
    In HX Stomp: create a path that starts at Return, add your effects

Keyboard shortcuts
Key 	Action
SPACE 	Play / Pause
← / → 	Seek ±5 seconds
ESC 	Quit
Click waveform 	Seek to position
Click marker button 	Jump to marker
Speed Trainer workflow

    Set A/B loop points around the section you're working on
    Enable LOOP
    Set TRAIN START (e.g. 60%), TRAIN TARGET (100%), STEP (5%), REPS (2)
    Press TRAIN — it auto-advances tempo every N loops

Phase 2 (coming next)

    Chord / key detection (librosa chroma)
    Beat/BPM detection (aubio)
    Stem separation (Demucs — offline preprocessing)
    Per-song session files
    Better EQ (parametric, not just shelves)

Dependencies

    pyrubberband — time/pitch stretching
    librosa — audio analysis
    soundfile — file I/O
    pygame — display + audio output
    scipy — EQ filters
    numpy — everything else
