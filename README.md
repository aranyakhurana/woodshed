# WOODSHED
### Guitar Practice Tool
*Raspberry Pi 5 + 7" touchscreen*

An app to help you learn difficult musical passages. Load any audio file, slow it down without changing pitch, mark up the structure, loop the tricky bits, and let the Speed Trainer gradually push you back up to tempo — one rep at a time.

---

## Features

| Feature | Details |
|---|---|
| **Speed** | 25–120% of original tempo, independent of pitch (Rubber Band library) |
| **Pitch** | ±12 semitones, independent of speed |
| **A/B Loop** | Set loop in and out points with `[ A` / `B ]`, toggle with LOOP |
| **Waveform** | Zoomable (1×–32×, snap to powers of 2), pannable, drag to select a loop region |
| **Markers** | Named markers placed anywhere on the timeline; tap to jump, scroll carousel when there are many |
| **Sections** | Save named A/B regions as reusable sections; displayed as labelled overlays on the waveform |
| **EQ** | Low / Mid / High shelf (±12 dB each), master on/off toggle |
| **Speed Trainer** | Auto-ramps tempo step by step; set start %, target %, step size, and reps per step |
| **Curve mode** | Non-linear trainer option — reps required scale up quadratically as speed increases, reinforcing the hardest passages |
| **Replay mode** | On play, snaps back to your last seek or marker jump position |
| **Session save/load** | Saves speed, pitch, EQ, loop points, markers, and sections to `~/woodshed_session.json` |
| **File browser** | Browse and open MP3, WAV, FLAC, AIFF, OGG, M4A |

---

## Setup

```bash
git clone https://github.com/aranyakhurana/woodshed.git
cd woodshed
bash src/setup.sh
```

---

## Running

```bash
# Open the file browser on launch
python3 src/woodshed.py

# With a file preloaded
python3 src/woodshed.py ~/Music/mysong.mp3
```

### Autostart on Pi boot (optional)

Create a systemd service or add to `/etc/rc.local` before `exit 0`:
```bash
DISPLAY=:0 python3 /home/pi/woodshed/src/woodshed.py &
```

---

## Controls

| Control | Action |
|---|---|
| `SPACE` | Play / Pause |
| `← / →` | Seek ±5 seconds |
| `ESC` | Quit |
| Click waveform | Seek to position |
| Drag waveform | Set A/B loop region |
| Click marker | Jump to marker position |
| Right-click marker | Rename or delete marker |
| Click section | Restore that A/B region |
| Right-click section | Rename or delete section |

---

## Speed Trainer workflow

1. Set A/B loop points around the passage you're working on
2. Click **SPEED TRAINER** to start — it sets speed to your chosen start %
3. The trainer advances tempo by the step amount every N loop repetitions
4. Enable **CURVE** for harder passages — reps required increase as speed rises
5. Click **SPEED TRAINER** again to stop

---

## Dependencies

- `pygame` — UI and display
- `sounddevice` — audio output
- `rubberband` (C library) — real-time time/pitch stretching
- `soundfile` — audio file I/O
- `numpy` — signal processing
- `scipy` — EQ filters
- `librosa` — audio loading and resampling
