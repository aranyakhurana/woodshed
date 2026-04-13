#!/usr/bin/env python3
"""
WOODSHED — Guitar Practice Tool
Phase 1: Speed, Pitch, Loop, Markers, EQ, Speed Trainer
Designed for Raspberry Pi 5 + 7" touchscreen
"""

import sys
import os
import math
import threading
import time
import json
import queue as _queue
import numpy as np
import soundfile as sf
import pygame
import sounddevice as sd
from pathlib import Path
from scipy.signal import sosfilt as _sosfilt
import ctypes, ctypes.util as _ctutil

# ─── Constants ────────────────────────────────────────────────────────────────

W, H = 848, 533   # set properly at runtime in WoodshedApp.__init__
FPS = 60

# Colour palette — dark industrial, amber accents
BG       = (10,  10,  12)
SURFACE  = (20,  20,  24)
CARD     = (28,  28,  34)
BORDER   = (45,  45,  55)
AMBER    = (255, 170,  40)
AMBER_DIM= (120,  80,  18)
PURPLE   = (155,  90, 215)
PURPLE_DIM = ( 85,  45, 130)
GREEN    = ( 80, 220, 120)
RED      = (220,  70,  70)
BLUE     = ( 80, 160, 255)
WHITE    = (240, 240, 240)
GREY     = (120, 120, 130)
DGREY    = ( 55,  55,  65)

FONT_MONO = None   # loaded at runtime
FONT_XS   = None
FONT_SM   = None
FONT_LG   = None
FONT_XL   = None

SAMPLE_RATE   = 44100
SD_BLOCK      = 1024              # sounddevice callback blocksize
RB_OUT_CHUNK  = SD_BLOCK * 4     # ~93 ms — max samples retrieved from stretcher per worker iteration

# ─── Rubberband library — direct C API for real-time streaming ────────────────

_RB_LIB_PATH = _ctutil.find_library('rubberband')
_librb = ctypes.CDLL(_RB_LIB_PATH) if _RB_LIB_PATH else None

if _librb:
    _vp = ctypes.c_void_p
    _librb.rubberband_new.restype          = _vp
    _librb.rubberband_new.argtypes         = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
                                               ctypes.c_double, ctypes.c_double]
    _librb.rubberband_delete.argtypes      = [_vp]
    _librb.rubberband_reset.argtypes       = [_vp]
    _librb.rubberband_set_time_ratio.argtypes  = [_vp, ctypes.c_double]
    _librb.rubberband_set_pitch_scale.argtypes = [_vp, ctypes.c_double]
    _librb.rubberband_get_latency.restype         = ctypes.c_uint
    _librb.rubberband_get_latency.argtypes        = [_vp]
    _librb.rubberband_get_samples_required.restype  = ctypes.c_uint
    _librb.rubberband_get_samples_required.argtypes = [_vp]
    _librb.rubberband_process.argtypes  = [_vp, ctypes.POINTER(_vp), ctypes.c_uint, ctypes.c_int]
    _librb.rubberband_available.restype  = ctypes.c_int
    _librb.rubberband_available.argtypes = [_vp]
    _librb.rubberband_retrieve.restype  = ctypes.c_uint
    _librb.rubberband_retrieve.argtypes = [_vp, ctypes.POINTER(_vp), ctypes.c_uint]

_RB_OPT_REALTIME = 0x00000001   # ProcessRealTime; EngineBuiltin (R2) is 0x0 so default

def _rb_new(sr, speed, pitch_semitones):
    if _librb is None:
        return None
    return _librb.rubberband_new(int(sr), 2, _RB_OPT_REALTIME,
                                  1.0 / max(speed, 1e-6),
                                  2.0 ** (pitch_semitones / 12.0))

def _rb_process(state, chunk, final=False):
    """Feed (N, 2) float32 stereo chunk into the stretcher."""
    ch0 = np.ascontiguousarray(chunk[:, 0], dtype=np.float32)
    ch1 = np.ascontiguousarray(chunk[:, 1], dtype=np.float32)
    ptrs = (ctypes.c_void_p * 2)(int(ch0.ctypes.data), int(ch1.ctypes.data))
    _librb.rubberband_process(state, ptrs, len(ch0), int(final))

def _rb_retrieve(state, n):
    """Retrieve up to n frames of stretched stereo audio. Returns (N, 2) float32."""
    out0 = np.empty(n, dtype=np.float32)
    out1 = np.empty(n, dtype=np.float32)
    ptrs = (ctypes.c_void_p * 2)(int(out0.ctypes.data), int(out1.ctypes.data))
    got = _librb.rubberband_retrieve(state, ptrs, n)
    if got == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([out0[:got], out1[:got]], axis=1)

# ─── Audio Engine ─────────────────────────────────────────────────────────────

class AudioEngine:
    def __init__(self):
        self.raw        = None      # original audio, float32 stereo [frames, 2]
        self.sr         = SAMPLE_RATE
        self.filepath   = None

        self._speed     = 1.0       # 0.25 – 2.0
        self._pitch     = 0         # semitones, -12 to +12
        self.volume     = 0.8       # 0.0 – 1.0
        self._eq_low    = 0.0       # dB -12 to +12
        self._eq_mid    = 0.0
        self._eq_high   = 0.0
        self._eq_enabled    = True  # master EQ on/off
        self._eq_sos        = None  # precomputed combined SOS (None = bypass)
        self._eq_sos_active = None  # SOS that was actually applied in the last callback block
        self._eq_zi         = None  # filter state (shape matches _eq_sos_active)
        self._eq_zi_reset   = False # True → callback crossfades to new _eq_sos on next block

        self._rb_state  = None      # rubberband stretcher (created on load, streaming C API)

        self.loop_a     = None      # raw sample index
        self.loop_b     = None
        self.loop_on    = False

        self.markers     = {}        # name -> raw sample index
        self.saved_sections = {}        # name -> (loop_a, loop_b) raw sample indices

        # Trainer
        self.trainer_on        = False
        self.trainer_nonlinear = False   # True → reps scale quadratically with speed
        self.trainer_start     = 0.50
        self.trainer_target    = 1.00
        self.trainer_step      = 0.05
        self.trainer_reps      = 2
        self._trainer_rep_count = 0
        self._trainer_current   = 0.50

        # Playback state — all positions in raw sample space
        self._raw_pos         = 0     # next chunk to process (may be ahead of playback)
        self._actual_pos      = 0     # start of the chunk currently being heard
        self._actual_pos_end  = 0     # end of the chunk currently being heard
        self._actual_pos_time = 0.0   # wall-clock time when that chunk was queued
        self._playing   = False
        self._lock      = threading.Lock()
        self._thread    = None

        # sounddevice / pre-processing queue
        self._pcm_queue    = _queue.Queue(maxsize=16)  # float32 (frames, 2) chunks
        self._sd_residual  = None                      # leftover frames from last chunk
        self._stream       = None
        self._params_changed = threading.Event()

    # ── Speed / Pitch / EQ properties (set _params_changed on change) ──────────

    @property
    def speed(self): return self._speed

    @speed.setter
    def speed(self, val):
        if val != self._speed:
            self._speed = val
            if self._rb_state is not None:
                _librb.rubberband_set_time_ratio(self._rb_state, 1.0 / max(val, 1e-6))

    @property
    def pitch(self): return self._pitch

    @pitch.setter
    def pitch(self, val):
        if val != self._pitch:
            self._pitch = val
            if self._rb_state is not None:
                _librb.rubberband_set_pitch_scale(self._rb_state, 2.0 ** (val / 12.0))

    @property
    def eq_low(self): return self._eq_low

    @eq_low.setter
    def eq_low(self, val):
        if val != self._eq_low:
            self._eq_low = val
            self._eq_sos = self._build_eq_sos()
            self._eq_zi_reset = True

    @property
    def eq_mid(self): return self._eq_mid

    @eq_mid.setter
    def eq_mid(self, val):
        if val != self._eq_mid:
            self._eq_mid = val
            self._eq_sos = self._build_eq_sos()
            self._eq_zi_reset = True

    @property
    def eq_high(self): return self._eq_high

    @eq_high.setter
    def eq_high(self, val):
        if val != self._eq_high:
            self._eq_high = val
            self._eq_sos = self._build_eq_sos()
            self._eq_zi_reset = True

    # ── File loading ──────────────────────────────────────────────────────────

    def load(self, path):
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data.T, orig_sr=sr, target_sr=SAMPLE_RATE).T
        self.raw      = data
        self.sr       = SAMPLE_RATE
        self.filepath = path
        self.loop_a   = None
        self.loop_b   = None
        self.markers  = {}
        self._raw_pos = 0
        self._playing = False
        # (Re-)create the rubberband streaming stretcher for this file
        if self._rb_state is not None and _librb:
            _librb.rubberband_delete(self._rb_state)
        self._rb_state = _rb_new(self.sr, self._speed, self._pitch)
        return True

    # ── EQ filter coefficient builders ───────────────────────────────────────

    def _sos_row(self, b0, b1, b2, a0, a1, a2):
        return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

    def _low_shelf_sos(self, f0, gain_db):
        A = 10 ** (gain_db / 40.0); w0 = 2*np.pi*f0/self.sr
        c, s = np.cos(w0), np.sin(w0); sqA = np.sqrt(A)
        alph = s * 0.7071067811865476
        return self._sos_row(
            A*((A+1)-(A-1)*c+2*sqA*alph), 2*A*((A-1)-(A+1)*c), A*((A+1)-(A-1)*c-2*sqA*alph),
              ((A+1)+(A-1)*c+2*sqA*alph), -2*((A-1)+(A+1)*c),    ((A+1)+(A-1)*c-2*sqA*alph))

    def _high_shelf_sos(self, f0, gain_db):
        A = 10 ** (gain_db / 40.0); w0 = 2*np.pi*f0/self.sr
        c, s = np.cos(w0), np.sin(w0); sqA = np.sqrt(A)
        alph = s * 0.7071067811865476
        return self._sos_row(
            A*((A+1)+(A-1)*c+2*sqA*alph), -2*A*((A-1)+(A+1)*c), A*((A+1)+(A-1)*c-2*sqA*alph),
              ((A+1)-(A-1)*c+2*sqA*alph),   2*((A-1)-(A+1)*c),    ((A+1)-(A-1)*c-2*sqA*alph))

    def _peak_eq_sos(self, f0, gain_db, Q=2.0):
        A = 10 ** (gain_db / 40.0); w0 = 2*np.pi*f0/self.sr
        alph = np.sin(w0)/(2*Q); c = np.cos(w0)
        return self._sos_row(1+alph*A, -2*c, 1-alph*A,  1+alph/A, -2*c, 1-alph/A)

    def _build_eq_sos(self):
        """Build combined SOS matrix for currently active EQ bands (None if bypassed)."""
        if not self._eq_enabled:
            return None
        parts = []
        if self._eq_low  != 0: parts.append(self._low_shelf_sos(300,  self._eq_low))
        if self._eq_mid  != 0: parts.append(self._peak_eq_sos(1000,   self._eq_mid))
        if self._eq_high != 0: parts.append(self._high_shelf_sos(4000, self._eq_high))
        return np.vstack(parts) if parts else None

    @property
    def eq_enabled(self): return self._eq_enabled

    @eq_enabled.setter
    def eq_enabled(self, val):
        if val != self._eq_enabled:
            self._eq_enabled = val
            self._eq_sos = self._build_eq_sos()
            self._eq_zi_reset = True

    # ── Processing ────────────────────────────────────────────────────────────

    def _apply_eq(self, audio):
        # Audio EQ Cookbook biquad shelf/peak filters (Robert Bristow-Johnson).
        # These work correctly for both boost and cut.
        from scipy.signal import sosfilt
        sr = self.sr

        def _sos(b0, b1, b2, a0, a1, a2):
            return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

        def low_shelf(data, f0, gain_db):
            A  = 10 ** (gain_db / 40.0)
            w0 = 2 * np.pi * f0 / sr
            c, s = np.cos(w0), np.sin(w0)
            sqA  = np.sqrt(A)
            alph = s * 0.7071067811865476   # sin(w0)/2 * sqrt(2), shelf slope S=1
            sos = _sos(
                 A*((A+1) - (A-1)*c + 2*sqA*alph),
               2*A*((A-1) - (A+1)*c),
                 A*((A+1) - (A-1)*c - 2*sqA*alph),
                   ((A+1) + (A-1)*c + 2*sqA*alph),
              -2  *((A-1) + (A+1)*c),
                   ((A+1) + (A-1)*c - 2*sqA*alph),
            )
            return sosfilt(sos, data, axis=0)

        def high_shelf(data, f0, gain_db):
            A  = 10 ** (gain_db / 40.0)
            w0 = 2 * np.pi * f0 / sr
            c, s = np.cos(w0), np.sin(w0)
            sqA  = np.sqrt(A)
            alph = s * 0.7071067811865476
            sos = _sos(
                 A*((A+1) + (A-1)*c + 2*sqA*alph),
              -2*A*((A-1) + (A+1)*c),
                 A*((A+1) + (A-1)*c - 2*sqA*alph),
                   ((A+1) - (A-1)*c + 2*sqA*alph),
               2  *((A-1) - (A+1)*c),
                   ((A+1) - (A-1)*c - 2*sqA*alph),
            )
            return sosfilt(sos, data, axis=0)

        def peak_eq(data, f0, gain_db, Q=2.0):
            A    = 10 ** (gain_db / 40.0)
            w0   = 2 * np.pi * f0 / sr
            alph = np.sin(w0) / (2 * Q)
            c    = np.cos(w0)
            sos = _sos(
                1 + alph*A,  -2*c,  1 - alph*A,
                1 + alph/A,  -2*c,  1 - alph/A,
            )
            return sosfilt(sos, data, axis=0)

        if self.eq_low  != 0: audio = low_shelf(audio,  300, self.eq_low)
        if self.eq_mid  != 0: audio = peak_eq(audio,   1000, self.eq_mid)
        if self.eq_high != 0: audio = high_shelf(audio, 4000, self.eq_high)
        return audio

    # ── Transport ─────────────────────────────────────────────────────────────

    # ── sounddevice stream ────────────────────────────────────────────────────

    def init_stream(self):
        """Create and start the persistent sounddevice output stream."""
        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype='float32',
                blocksize=SD_BLOCK,
                callback=self._sd_callback,
            )
            self._stream.start()
        except Exception as exc:
            print(f'sounddevice init error: {exc}', file=sys.stderr)

    def _sd_callback(self, outdata, frames, time_info, status):
        """sounddevice real-time callback — reads pre-processed audio from queue."""
        if not self._playing:
            outdata[:] = 0
            return
        filled = 0
        while filled < frames:
            if self._sd_residual is not None:
                n = min(frames - filled, len(self._sd_residual))
                outdata[filled:filled + n] = self._sd_residual[:n]
                filled += n
                rem = self._sd_residual[n:]
                self._sd_residual = rem if len(rem) else None
            else:
                try:
                    chunk, raw_pos, raw_pos_end = self._pcm_queue.get_nowait()
                    self._actual_pos      = raw_pos
                    self._actual_pos_end  = raw_pos_end
                    self._actual_pos_time = time.monotonic()
                    self._sd_residual = chunk
                except _queue.Empty:
                    break
        if filled > 0:
            target_sos = self._eq_sos
            if self._eq_zi_reset:
                self._eq_zi_reset = False
                old_sos = self._eq_sos_active
                old_zi  = self._eq_zi
                # Compute "old" output (finish outgoing filter state)
                if old_sos is not None and old_zi is not None:
                    y0_old, _ = _sosfilt(old_sos, outdata[:filled, 0], zi=old_zi[:, 0, :])
                    y1_old, _ = _sosfilt(old_sos, outdata[:filled, 1], zi=old_zi[:, 1, :])
                else:
                    y0_old = outdata[:filled, 0].copy()
                    y1_old = outdata[:filled, 1].copy()
                # Compute "new" output (cold-start incoming filter)
                if target_sos is not None:
                    new_zi = np.zeros((target_sos.shape[0], 2, 2), dtype=np.float64)
                    y0_new, new_zi[:, 0, :] = _sosfilt(target_sos, outdata[:filled, 0], zi=new_zi[:, 0, :])
                    y1_new, new_zi[:, 1, :] = _sosfilt(target_sos, outdata[:filled, 1], zi=new_zi[:, 1, :])
                    self._eq_zi = new_zi
                else:
                    y0_new = outdata[:filled, 0].copy()
                    y1_new = outdata[:filled, 1].copy()
                    self._eq_zi = None
                # Linear crossfade old → new over this block
                fade = np.linspace(0.0, 1.0, filled, dtype=np.float32)
                outdata[:filled, 0] = (y0_old * (1.0 - fade) + y0_new * fade).astype(np.float32)
                outdata[:filled, 1] = (y1_old * (1.0 - fade) + y1_new * fade).astype(np.float32)
                self._eq_sos_active = target_sos
            elif target_sos is not None:
                # Steady-state: apply filter with maintained state
                zi = self._eq_zi
                if zi is None or zi.shape[0] != target_sos.shape[0]:
                    zi = np.zeros((target_sos.shape[0], 2, 2), dtype=np.float64)
                y0, zi[:, 0, :] = _sosfilt(target_sos, outdata[:filled, 0], zi=zi[:, 0, :])
                y1, zi[:, 1, :] = _sosfilt(target_sos, outdata[:filled, 1], zi=zi[:, 1, :])
                outdata[:filled, 0] = y0
                outdata[:filled, 1] = y1
                self._eq_zi = zi
                self._eq_sos_active = target_sos
            outdata[:filled] *= self.volume
        if filled < frames:
            outdata[filled:] = 0

    def play(self):
        if self.raw is None:
            return
        # Discard any stale pre-processed audio
        while True:
            try: self._pcm_queue.get_nowait()
            except _queue.Empty: break
        self._sd_residual = None
        self._playing = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def pause(self):
        # Snapshot the interpolated playback position while _playing is still True
        # so the interpolation formula is active.  After this we reset _raw_pos so
        # that resume (and REPLAY-off mode) starts from exactly here.
        if self.raw is not None:
            snap = max(0, min(int(self.position_seconds * self.sr), len(self.raw) - 1))
        else:
            snap = None

        self._playing = False

        while True:
            try: self._pcm_queue.get_nowait()
            except _queue.Empty: break
        self._sd_residual = None

        if snap is not None:
            with self._lock:
                self._raw_pos         = snap
                self._actual_pos      = snap
                self._actual_pos_time = 0.0

    def toggle(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def _flush_buffers(self):
        """Drain pre-buffered audio so the callback won't play stale chunks after a seek."""
        while True:
            try: self._pcm_queue.get_nowait()
            except _queue.Empty: break
        self._sd_residual = None

    def seek(self, fraction):
        if self.raw is not None:
            with self._lock:
                self._raw_pos = int(fraction * len(self.raw))
                self._actual_pos = self._raw_pos
                self._actual_pos_time = 0.0
            self._flush_buffers()
            self._params_changed.set()

    def seek_seconds(self, secs):
        if self.raw is not None:
            with self._lock:
                self._raw_pos = max(0, min(int(secs * self.sr), len(self.raw) - 1))
                self._actual_pos = self._raw_pos
                self._actual_pos_time = 0.0
            self._flush_buffers()
            self._params_changed.set()

    @property
    def position_seconds(self):
        """Smooth position: interpolates between chunk boundaries using wall clock."""
        base = self._actual_pos
        t    = self._actual_pos_time
        if self._playing and t > 0:
            elapsed  = time.monotonic() - t
            # advance at the raw-sample rate scaled by playback speed
            estimated = base + elapsed * self.sr * self.speed
            # Clamp to the end of the chunk currently being heard — this is always
            # a reliable upper bound regardless of whether the worker has looped.
            estimated = min(estimated, self._actual_pos_end)
            return estimated / self.sr
        return base / self.sr

    @property
    def position_fraction(self):
        if self.raw is None or len(self.raw) == 0:
            return 0
        return self.position_seconds * self.sr / len(self.raw)

    @property
    def duration_seconds(self):
        if self.raw is None:
            return 0
        return len(self.raw) / self.sr

    @property
    def raw_duration_seconds(self):
        return self.duration_seconds

    def _worker_loop(self):
        """Background thread: feeds raw audio through the rubberband streaming stretcher."""
        # Fresh start — reset stretcher so it has no stale context
        if self._rb_state is not None:
            _librb.rubberband_reset(self._rb_state)

        _accum   = []   # accumulator: list of (N,2) float32 arrays
        _accum_n = 0    # total frames accumulated
        _ACCUM_MIN = SD_BLOCK * 2  # don't queue until we have at least 2048 samples

        while self._playing:
            # ── Seek / position reset ─────────────────────────────────────────
            if self._params_changed.is_set():
                self._params_changed.clear()
                _accum   = []
                _accum_n = 0
                if self.raw is not None:
                    new_pos = max(0, min(int(self.position_seconds * self.sr), len(self.raw) - 1))
                while True:
                    try: self._pcm_queue.get_nowait()
                    except _queue.Empty: break
                self._sd_residual = None
                if self._rb_state is not None:
                    _librb.rubberband_reset(self._rb_state)
                if self.raw is not None:
                    with self._lock:
                        self._raw_pos         = new_pos
                        self._actual_pos      = new_pos
                        self._actual_pos_end  = new_pos
                        self._actual_pos_time = 0.0
                continue

            if self.raw is None or self._rb_state is None:
                time.sleep(0.02)
                continue

            with self._lock:
                raw_pos = self._raw_pos

            loop_on   = self.loop_on
            la, lb    = self.loop_a, self.loop_b
            raw_end   = lb if (loop_on and lb is not None) else len(self.raw)
            raw_start = la if (loop_on and la is not None) else 0

            # ── Feed raw input into stretcher ─────────────────────────────────
            samples_req = int(_librb.rubberband_get_samples_required(self._rb_state))
            if samples_req > 0:
                if raw_pos >= raw_end:
                    # End of region — handle loop or stop
                    if loop_on and la is not None:
                        if self.trainer_on:
                            # Non-linear mode: reps needed scale quadratically with speed
                            # so passages at higher speeds demand more repetitions
                            if self.trainer_nonlinear and self.trainer_start > 0:
                                ratio = self._trainer_current / self.trainer_start
                                eff_reps = max(1, round(self.trainer_reps * ratio * ratio))
                            else:
                                eff_reps = self.trainer_reps
                            self._trainer_rep_count += 1
                            if self._trainer_rep_count >= eff_reps:
                                self._trainer_rep_count = 0
                                new_speed = min(
                                    self._trainer_current + self.trainer_step,
                                    self.trainer_target,
                                )
                                if new_speed != self._trainer_current:
                                    self._trainer_current = new_speed
                                    self.speed = new_speed   # uses setter → updates stretcher live
                        with self._lock:
                            self._raw_pos = raw_start
                        continue
                    else:
                        # Past end with no loop — drain any remaining output then stop
                        if _librb.rubberband_available(self._rb_state) <= 0:
                            # Flush any accumulated audio before stopping
                            if _accum_n > 0:
                                combined = np.concatenate(_accum) if len(_accum) > 1 else _accum[0]
                                _accum = []; _accum_n = 0
                                with self._lock:
                                    cur_pos = self._raw_pos
                                try:
                                    self._pcm_queue.put((combined, cur_pos, cur_pos), timeout=0.1)
                                except _queue.Full:
                                    pass
                            self._playing = False
                            break
                        # Fall through to collect remaining output below
                else:
                    feed_n = min(samples_req, raw_end - raw_pos)
                    _rb_process(self._rb_state, self.raw[raw_pos:raw_pos + feed_n])
                    with self._lock:
                        self._raw_pos = raw_pos + feed_n

            # ── Collect output from stretcher ─────────────────────────────────
            avail = _librb.rubberband_available(self._rb_state)
            if avail <= 0:
                time.sleep(0.001)
                continue

            retrieve_n = min(int(avail), RB_OUT_CHUNK)
            out = _rb_retrieve(self._rb_state, retrieve_n)
            if len(out) == 0:
                continue

            out = np.clip(out, -1.0, 1.0).astype(np.float32)

            _accum.append(out)
            _accum_n += len(out)

            if _accum_n < _ACCUM_MIN:
                continue   # keep accumulating before queuing

            combined = np.concatenate(_accum) if len(_accum) > 1 else _accum[0]
            _accum   = []
            _accum_n = 0

            with self._lock:
                cur_pos = self._raw_pos

            # Block until queue has room, waking early on seek
            while self._playing and not self._params_changed.is_set():
                try:
                    self._pcm_queue.put((combined, cur_pos, cur_pos), timeout=0.05)
                    break
                except _queue.Full:
                    continue

    # ── Markers ───────────────────────────────────────────────────────────────

    def set_marker(self, name):
        self.markers[name] = self._raw_pos

    def goto_marker(self, name):
        if name in self.markers:
            self._jump_to_pos(self.markers[name])

    def goto_next_marker(self):
        if not self.markers or self.raw is None:
            return
        pos  = self._raw_pos
        after = [idx for idx in self.markers.values() if idx > pos]
        self._jump_to_pos(min(after) if after else min(self.markers.values()))

    def goto_prev_marker(self):
        if not self.markers or self.raw is None:
            return
        pos    = self._raw_pos
        before = [idx for idx in self.markers.values() if idx < pos]
        self._jump_to_pos(max(before) if before else max(self.markers.values()))

    def _jump_to_pos(self, idx):
        with self._lock:
            self._raw_pos         = idx
            self._actual_pos      = idx
            self._actual_pos_time = 0.0
        self._flush_buffers()
        self._params_changed.set()

    def set_loop_a(self):
        self.loop_a = self._actual_pos

    def set_loop_b(self):
        self.loop_b = self._actual_pos

    def clear_loop(self):
        self.loop_a = None
        self.loop_b = None
        self.loop_on = False

    # ── Session save/load ─────────────────────────────────────────────────────

    def save_session(self, path):
        session = {
            'filepath':      str(self.filepath),
            'speed':         self.speed,
            'pitch':         self.pitch,
            'volume':        self.volume,
            'eq_low':        self.eq_low,
            'eq_mid':        self.eq_mid,
            'eq_high':       self.eq_high,
            'loop_a':        self.loop_a,
            'loop_b':        self.loop_b,
            'loop_on':       self.loop_on,
            'markers':       self.markers,
            'saved_sections':   self.saved_sections,
            'trainer_start':     self.trainer_start,
            'trainer_target':    self.trainer_target,
            'trainer_step':      self.trainer_step,
            'trainer_reps':      self.trainer_reps,
            'trainer_nonlinear': self.trainer_nonlinear,
        }
        with open(path, 'w') as f:
            json.dump(session, f, indent=2)

    def load_session(self, path):
        with open(path) as f:
            s = json.load(f)
        self.load(s['filepath'])
        self.speed   = s.get('speed',  1.0)
        self.pitch   = s.get('pitch',  0)
        self.volume  = s.get('volume', 0.8)
        self.eq_low  = s.get('eq_low', 0)
        self.eq_mid  = s.get('eq_mid', 0)
        self.eq_high = s.get('eq_high', 0)
        self.loop_a  = s.get('loop_a')
        self.loop_b  = s.get('loop_b')
        self.loop_on = s.get('loop_on', False)
        self.markers      = s.get('markers', {})
        raw_sections = s.get('saved_sections', {})
        # stored as lists in JSON — convert back to tuples
        self.saved_sections = {k: tuple(v) for k, v in raw_sections.items()}
        self.trainer_start     = s.get('trainer_start',     0.5)
        self.trainer_target    = s.get('trainer_target',    1.0)
        self.trainer_step      = s.get('trainer_step',      0.05)
        self.trainer_reps      = s.get('trainer_reps',      2)
        self.trainer_nonlinear = s.get('trainer_nonlinear', False)

    def start_trainer(self):
        self._trainer_current   = self.trainer_start
        self._trainer_rep_count = 0
        self.speed      = self.trainer_start
        self.trainer_on = True
        self.loop_on    = True

    def stop_trainer(self):
        self.trainer_on = False


# ─── UI Helpers ───────────────────────────────────────────────────────────────

def draw_rect(surf, color, rect, radius=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

def draw_text(surf, text, font, color, x, y, anchor='topleft'):
    img = font.render(str(text), True, color)
    r = img.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(img, r)
    return r

def fmt_time(secs):
    m = int(secs) // 60
    s = int(secs) % 60
    cs = int((secs - int(secs)) * 10)
    return f"{m}:{s:02d}.{cs}"


# ─── Slider Widget ────────────────────────────────────────────────────────────

_SL_BTN_W = 26   # width of each −/+ button flanking the track

class Slider:
    def __init__(self, rect, vmin, vmax, value, label, fmt='{:.2f}', color=AMBER,
                 step=None, default=None, toggleable=False):
        self.rect    = pygame.Rect(rect)
        self.vmin    = vmin
        self.vmax    = vmax
        self.value   = value
        self.label   = label
        self.fmt     = fmt
        self.color   = color
        self.step    = step if step is not None else (vmax - vmin) / 20
        self.default = default if default is not None else value
        self.dragging  = False
        self.toggleable = toggleable
        self.enabled    = True    # when False the effect is bypassed; label tap toggles this
        self._val_rect   = None   # set during draw; tapping resets to default
        self._label_rect = None   # set during draw; tapping toggles enabled

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _track_rect(self):
        """Horizontal track region (inset from the full rect to leave room for buttons)."""
        r = self.rect
        bw = _SL_BTN_W
        return pygame.Rect(r.x + bw + 4, r.y, r.width - 2 * (bw + 4), r.height)

    def _btn_rects(self):
        """Returns (minus_rect, plus_rect) for the −/+ increment buttons."""
        r = self.rect
        bw = _SL_BTN_W
        # Make buttons a bit taller than the track for easier tapping
        bh = r.height + 16
        by = r.centery - bh // 2
        minus = pygame.Rect(r.x, by, bw, bh)
        plus  = pygame.Rect(r.right - bw, by, bw, bh)
        return minus, plus

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surf, fonts):
        r  = self.rect
        tr = self._track_rect()
        bm, bp = self._btn_rects()
        thumb_r = max(8, r.height + 4)

        # −/+ buttons
        draw_rect(surf, DGREY, bm, radius=4)
        draw_rect(surf, DGREY, bp, radius=4)
        draw_text(surf, '−', fonts['sm'], WHITE, bm.centerx, bm.centery, 'center')
        draw_text(surf, '+', fonts['sm'], WHITE, bp.centerx, bp.centery, 'center')

        # Track — dimmed when bypassed
        active_color = self.color if self.enabled else DGREY
        track = pygame.Rect(tr.x, tr.centery - 3, tr.width, 6)
        draw_rect(surf, DGREY, track, radius=3)
        frac = (self.value - self.vmin) / (self.vmax - self.vmin)
        fill = pygame.Rect(tr.x, tr.centery - 3, int(tr.width * frac), 6)
        draw_rect(surf, active_color, fill, radius=3)

        # Thumb
        tx = tr.x + int(tr.width * frac)
        pygame.draw.circle(surf, active_color, (tx, tr.centery), thumb_r)
        pygame.draw.circle(surf, BG, (tx, tr.centery), max(3, thumb_r - 5))

        # Label + value — above the thumb row
        label_y = r.y - thumb_r - 4

        if self.toggleable:
            # Label is a toggle indicator: dot + text, tappable
            dot_col  = GREEN if self.enabled else DGREY
            lbl_col  = WHITE if self.enabled else GREY
            dot_x    = tr.x + 6
            dot_y    = label_y + fonts['sm'].get_height() // 2
            pygame.draw.circle(surf, dot_col, (dot_x, dot_y), 4)
            lr = draw_text(surf, self.label, fonts['sm'], lbl_col, tr.x + 14, label_y, 'topleft')
            self._label_rect = pygame.Rect(tr.x, label_y - 2, lr.right - tr.x + 6, lr.height + 4)
        else:
            draw_text(surf, self.label, fonts['sm'], GREY, tr.x, label_y, 'topleft')
            self._label_rect = None

        # Value: amber when non-default (tap to reset), white when at default
        at_default = abs(self.value - self.default) < 1e-9
        val_color  = WHITE if at_default else AMBER
        val_str    = self.fmt.format(self.value)
        vr = draw_text(surf, val_str, fonts['sm'], val_color, tr.right, label_y, 'topright')
        self._val_rect = vr.inflate(8, 6)   # slightly larger hit-area for tap-to-reset

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, event):
        tr = self._track_rect()
        bm, bp = self._btn_rects()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            # Tap on label: toggle enabled (toggleable sliders only)
            if self.toggleable and self._label_rect and self._label_rect.collidepoint(pos):
                self.enabled = not self.enabled
                return True

            # −/+ buttons: step increment
            if bm.collidepoint(pos):
                self.value = max(self.vmin, self.value - self.step)
                return True
            if bp.collidepoint(pos):
                self.value = min(self.vmax, self.value + self.step)
                return True

            # Tap on value text: reset to default
            if self._val_rect and self._val_rect.collidepoint(pos):
                self.value = self.default
                return True

            # Track drag
            if tr.collidepoint(pos):
                self.dragging = True
                self._update_track(pos[0], tr)
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_track(event.pos[0], self._track_rect())
                return True

        return False

    def _update_track(self, x, tr):
        frac = (x - tr.x) / tr.width
        frac = max(0.0, min(1.0, frac))
        self.value = self.vmin + frac * (self.vmax - self.vmin)


# ─── Button Widget ────────────────────────────────────────────────────────────

class Button:
    def __init__(self, rect, label, color=DGREY, text_color=WHITE, toggle=False, active=False,
                 active_color=AMBER):
        self.rect         = pygame.Rect(rect)
        self.label        = label
        self.color        = color
        self.text_color   = text_color
        self.toggle       = toggle
        self.active       = active
        self.pressed      = False
        self.active_color = active_color

    def draw(self, surf, fonts):
        col = self.color if not self.active else self.active_color
        draw_rect(surf, col, self.rect, radius=6, border=1, border_color=BORDER)
        tc = BG if self.active else self.text_color
        draw_text(surf, self.label, fonts['sm'], tc,
                  self.rect.centerx, self.rect.centery, 'center')

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                if self.toggle:
                    self.active = not self.active
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False
        return False


# ─── Waveform Widget ─────────────────────────────────────────────────────────

class Waveform:
    MAX_ZOOM = 32.0

    def __init__(self, rect):
        self.rect       = pygame.Rect(rect)
        self.surface    = None   # pre-rendered full-song surface
        self.zoom       = 1.0   # 1 = full song visible; 2 = half; etc.
        self.view_start = 0.0   # fraction of song at left edge of viewport
        self._drag_x    = None  # mouse x at drag start
        self._sel_start = None  # song fraction where drag started
        self._sel_end   = None  # song fraction at current drag position
        self._pan_x     = None  # mouse x at right-click pan start

    @property
    def view_end(self):
        return min(1.0, self.view_start + 1.0 / self.zoom)

    # Power-of-2 zoom levels: 1, 2, 4, 8, 16, 32
    _ZOOM_LEVELS = [2**i for i in range(int(math.log2(32)) + 1)]  # [1,2,4,8,16,32]

    def _snap_zoom_up(self):
        """Return the next power-of-2 zoom level strictly above self.zoom."""
        for lvl in self._ZOOM_LEVELS:
            if lvl > self.zoom + 1e-9:
                return min(self.MAX_ZOOM, lvl)
        return self.MAX_ZOOM

    def _snap_zoom_down(self):
        """Return the next power-of-2 zoom level strictly below self.zoom."""
        for lvl in reversed(self._ZOOM_LEVELS):
            if lvl < self.zoom - 1e-9:
                return max(1.0, lvl)
        return 1.0

    def zoom_in(self, center_frac=None):
        if self.zoom >= self.MAX_ZOOM:
            return
        if center_frac is None:
            center_frac = self.view_start + 0.5 / self.zoom
        new_zoom = self._snap_zoom_up()
        visible = 1.0 / new_zoom
        self.view_start = max(0.0, min(1.0 - visible, center_frac - visible * 0.5))
        self.zoom = new_zoom

    def zoom_out(self):
        new_zoom = self._snap_zoom_down()
        center_frac = self.view_start + 0.5 / self.zoom
        self.zoom = new_zoom
        if self.zoom == 1.0:
            self.view_start = 0.0
        else:
            visible = 1.0 / self.zoom
            self.view_start = max(0.0, min(1.0 - visible, center_frac - visible * 0.5))

    def zoom_to_fit(self, a_frac, b_frac):
        """Zoom to show a_frac..b_frac centred in the view with a small margin."""
        lo, hi = min(a_frac, b_frac), max(a_frac, b_frac)
        span = max(hi - lo, 1.0 / self.MAX_ZOOM)
        # 10% margin on each side
        new_zoom = min(self.MAX_ZOOM, 0.8 / span)
        self.zoom = new_zoom
        visible = 1.0 / new_zoom
        center = (lo + hi) / 2
        self.view_start = max(0.0, min(1.0 - visible, center - visible / 2))

    def pan(self, delta_frac):
        visible = 1.0 / self.zoom
        self.view_start = max(0.0, min(1.0 - visible, self.view_start + delta_frac))

    def follow_playhead(self, pos_frac):
        """Scroll view to keep pos_frac visible when playing."""
        if self.zoom <= 1.0:
            return
        if not (self.view_start <= pos_frac <= self.view_end):
            visible = 1.0 / self.zoom
            self.view_start = max(0.0, min(1.0 - visible, pos_frac - visible * 0.1))

    def build(self, audio, color=PURPLE):
        w, h = self.rect.width, self.rect.height
        # Build at MAX_ZOOM × display width so each bar stays 1 px wide at any zoom level.
        surf_w = w * int(self.MAX_ZOOM)
        surf = pygame.Surface((surf_w, h))
        surf.fill(CARD)
        if audio is None or len(audio) == 0:
            self.surface = surf
            return
        mono = audio[:, 0] if audio.ndim == 2 else audio
        n = len(mono)
        step = max(1, n // surf_w)
        # Vectorised peak extraction
        trunc = (n // step) * step
        chunks = mono[:trunc].reshape(-1, step)
        peaks = np.max(np.abs(chunks), axis=1)
        if len(peaks) < surf_w:
            peaks = np.pad(peaks, (0, surf_w - len(peaks)))
        else:
            peaks = peaks[:surf_w]
        cy = h // 2
        for x, p in enumerate(peaks):
            ph = int(float(p) * cy * 0.9)
            pygame.draw.line(surf, color, (x, cy - ph), (x, cy + ph))
        self.surface = surf

    def _to_screen(self, frac):
        """Convert absolute song fraction to pixel x on screen."""
        r = self.rect
        vf = (frac - self.view_start) * self.zoom   # 0..1 within viewport
        return r.x + int(vf * r.width)

    def _in_view(self, frac):
        return self.view_start <= frac <= self.view_end

    def draw(self, surf, pos_frac, loop_a_frac=None, loop_b_frac=None, markers=None, duration=0,
             jump_frac=None, loop_active=False, saved_sections=None, sr=SAMPLE_RATE):
        r = self.rect
        draw_rect(surf, CARD, r, radius=4, border=1, border_color=BORDER)

        if self.surface:
            # Always scale from the high-res surface so bars stay 1 px at any zoom level
            sw = self.surface.get_width()
            src_x = int(self.view_start * sw)
            src_w = max(1, int(sw / self.zoom))
            src_rect = pygame.Rect(src_x, 0, src_w, self.surface.get_height())
            try:
                sub = self.surface.subsurface(src_rect)
                scaled = pygame.transform.scale(sub, (r.width, r.height))
                surf.blit(scaled, r.topleft)
            except ValueError:
                pass

        # Saved sections — drawn behind everything else
        if saved_sections and duration > 0 and sr > 0:
            for sec_name, (sec_a, sec_b) in saved_sections.items():
                a_frac = (sec_a / sr) / duration
                b_frac = (sec_b / sr) / duration
                av = max(self.view_start, a_frac)
                bv = min(self.view_end,   b_frac)
                if bv > av:
                    ax = self._to_screen(av)
                    bx = self._to_screen(bv)
                    shade = pygame.Surface((max(1, bx - ax), r.height), pygame.SRCALPHA)
                    shade.fill((100, 55, 180, 38))
                    surf.blit(shade, (ax, r.y))
                # Label: centred over the visible portion
                if a_frac < self.view_end and b_frac > self.view_start and FONT_XS is not None:
                    vis_ax = self._to_screen(max(a_frac, self.view_start))
                    vis_bx = self._to_screen(min(b_frac, self.view_end))
                    cx = (vis_ax + vis_bx) // 2
                    lbl = FONT_XS.render(sec_name[:14], True, PURPLE_DIM)
                    surf.blit(lbl, (cx - lbl.get_width() // 2, r.y + 3))

        # Loop region — A and B lines draw independently; shade only when both set
        if loop_a_frac is not None and loop_b_frac is not None:
            la_v = max(self.view_start, loop_a_frac)
            lb_v = min(self.view_end,   loop_b_frac)
            if lb_v > la_v:
                lax = self._to_screen(la_v)
                lbx = self._to_screen(lb_v)
                shade = pygame.Surface((max(1, lbx - lax), r.height), pygame.SRCALPHA)
                shade.fill((150, 80, 220, 55) if loop_active else (95, 60, 130, 42))
                surf.blit(shade, (lax, r.y))
        if loop_a_frac is not None and self._in_view(loop_a_frac):
            pygame.draw.line(surf, GREEN, (self._to_screen(loop_a_frac), r.y),
                             (self._to_screen(loop_a_frac), r.bottom), 2)
        if loop_b_frac is not None and self._in_view(loop_b_frac):
            pygame.draw.line(surf, RED,   (self._to_screen(loop_b_frac), r.y),
                             (self._to_screen(loop_b_frac), r.bottom), 2)

        # Drag selection highlight
        if self._sel_start is not None and self._sel_end is not None:
            a = min(self._sel_start, self._sel_end)
            b = max(self._sel_start, self._sel_end)
            av = max(self.view_start, a)
            bv = min(self.view_end,   b)
            if bv > av:
                ax = self._to_screen(av)
                bx = self._to_screen(bv)
                shade = pygame.Surface((max(1, bx - ax), r.height), pygame.SRCALPHA)
                shade.fill((160, 100, 255, 55))
                surf.blit(shade, (ax, r.y))
            if self._in_view(self._sel_start):
                pygame.draw.line(surf, PURPLE, (self._to_screen(self._sel_start), r.y),
                                 (self._to_screen(self._sel_start), r.bottom), 1)
            if self._in_view(self._sel_end):
                pygame.draw.line(surf, PURPLE, (self._to_screen(self._sel_end), r.y),
                                 (self._to_screen(self._sel_end), r.bottom), 1)

        # Markers — sorted by position, with tiny rotated name label
        if markers and duration > 0 and FONT_SM is not None:
            for name, idx in sorted(markers.items(), key=lambda x: x[1]):
                frac = (idx / SAMPLE_RATE) / duration
                if self._in_view(frac):
                    mx = self._to_screen(frac)
                    pygame.draw.line(surf, BLUE, (mx, r.y), (mx, r.bottom), 1)
                    if FONT_XS is not None:
                        lbl = FONT_XS.render(name[:10], True, BLUE)
                        surf.blit(lbl, (mx + 5, r.y + 2))

        # Jump-position cursor (blinks like a text-editor caret)
        if jump_frac is not None and self._in_view(jump_frac):
            if int(time.monotonic() * 2) % 2 == 0:   # 2 Hz blink
                jx = self._to_screen(jump_frac)
                # Dashed vertical line: draw alternating segments every 4 px
                y = r.y
                while y < r.bottom:
                    seg_end = min(y + 3, r.bottom)
                    pygame.draw.line(surf, (95, 95, 95), (jx, y), (jx, seg_end), 1)
                    y += 6
                # Small arrow heads to look like a cursor
                pygame.draw.polygon(surf, (95, 95, 95),
                                    [(jx-4, r.y), (jx+4, r.y), (jx, r.y+5)])
                pygame.draw.polygon(surf, (95, 95, 95),
                                    [(jx-4, r.bottom), (jx+4, r.bottom), (jx, r.bottom-5)])

        # Playhead
        if self._in_view(pos_frac):
            pygame.draw.line(surf, WHITE, (self._to_screen(pos_frac), r.y),
                             (self._to_screen(pos_frac), r.bottom), 2)


    def _x_to_frac(self, screen_x):
        """Convert a screen x pixel to an absolute song fraction."""
        r = self.rect
        widget_frac = (screen_x - r.x) / r.width
        return max(0.0, min(1.0, self.view_start + widget_frac / self.zoom))

    def handle_event(self, event, zoom_in_rect=None, zoom_out_rect=None, extra_rects=()):
        """Returns:
          - float            : user clicked (no drag) → seek to this fraction
          - (float, float)   : user dragged → set loop (a_frac, b_frac)
          - None             : no action
        Also handles mouse-wheel zoom."""
        r = self.rect

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if zoom_in_rect and zoom_in_rect.collidepoint(pos):
                return None
            if zoom_out_rect and zoom_out_rect.collidepoint(pos):
                return None
            for er in extra_rects:
                if er and er.collidepoint(pos):
                    return None
            if r.collidepoint(pos):
                if event.button == 1:
                    frac = self._x_to_frac(pos[0])
                    self._drag_x    = pos[0]
                    self._sel_start = frac
                    self._sel_end   = None
                elif event.button == 3:   # right-click = pan
                    self._pan_x = pos[0]
                elif event.button == 4:   # scroll up = zoom in
                    self.zoom_in(self._x_to_frac(pos[0]))
                elif event.button == 5:   # scroll down = zoom out
                    self.zoom_out()

        elif event.type == pygame.MOUSEMOTION:
            if self._drag_x is not None and self._sel_start is not None:
                if abs(event.pos[0] - self._drag_x) > 5:
                    self._sel_end = self._x_to_frac(event.pos[0])
            if self._pan_x is not None:
                dx = event.pos[0] - self._pan_x
                if abs(dx) > 2:
                    self.pan(-dx / (self.rect.width * self.zoom))
                    self._pan_x = event.pos[0]

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                self._pan_x = None
            if event.button == 1:
                sel_start = self._sel_start
                sel_end   = self._sel_end
                self._drag_x    = None
                self._sel_start = None
                self._sel_end   = None
                if sel_end is not None:
                    a = min(sel_start, sel_end)
                    b = max(sel_start, sel_end)
                    return (a, b)
                elif sel_start is not None:
                    return sel_start

        return None

    def handle_click(self, event):
        """Legacy simple click → seek fraction (no zoom awareness). Use handle_event instead."""
        return self.handle_event(event)


# ─── File Browser ─────────────────────────────────────────────────────────────

class FileBrowser:
    EXTENSIONS = {'.mp3', '.wav', '.flac', '.aiff', '.ogg', '.m4a'}

    def __init__(self, start_dir=None):
        self.root    = Path(start_dir or Path.home() / 'Music')
        self.current = self.root
        self.entries = []
        self.scroll  = 0
        self.visible = False
        self._scan()

    def _scan(self):
        try:
            items = sorted(self.current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            self.entries = [p for p in items
                            if p.is_dir() or p.suffix.lower() in self.EXTENSIONS]
        except Exception:
            self.entries = []
        self.scroll = 0

    # Scroll arrow button rects (set during draw, read in handle_event)
    _scroll_up_r = None
    _scroll_dn_r = None
    _close_r     = None

    def draw(self, surf, fonts):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        surf.blit(overlay, (0, 0))

        panel    = pygame.Rect(30, 30, W - 60, H - 60)
        btn_w    = max(60, int(panel.width * 0.08))
        row_h    = max(44, int(panel.height * 0.09))
        header_h = row_h + 4
        list_top = panel.y + header_h + row_h

        draw_rect(surf, SURFACE, panel, radius=8, border=1, border_color=AMBER)

        draw_text(surf, 'OPEN FILE', fonts['lg'], AMBER, panel.x + 12, panel.y + 10, 'topleft')
        FileBrowser._close_r = pygame.Rect(panel.right - btn_w - 6, panel.y + 6, btn_w, row_h - 4)
        draw_rect(surf, RED, FileBrowser._close_r, radius=4)
        draw_text(surf, '✕', fonts['lg'], WHITE, FileBrowser._close_r.centerx, FileBrowser._close_r.centery, 'center')

        draw_text(surf, str(self.current), fonts['sm'], GREY, panel.x + 12, panel.y + header_h, 'topleft')

        arrow_w = btn_w
        list_h  = panel.bottom - 8 - list_top
        FileBrowser._scroll_up_r = pygame.Rect(panel.right - arrow_w - 4, list_top,               arrow_w, list_h // 2 - 2)
        FileBrowser._scroll_dn_r = pygame.Rect(panel.right - arrow_w - 4, list_top + list_h // 2 + 2, arrow_w, list_h // 2 - 2)

        up_col = AMBER if self.scroll > 0 else DGREY
        dn_col = AMBER if self.scroll + self._visible_rows(row_h, list_h) < len(self.entries) else DGREY
        draw_rect(surf, up_col, FileBrowser._scroll_up_r, radius=4)
        draw_rect(surf, dn_col, FileBrowser._scroll_dn_r, radius=4)
        draw_text(surf, '▲', fonts['lg'], WHITE, FileBrowser._scroll_up_r.centerx, FileBrowser._scroll_up_r.centery, 'center')
        draw_text(surf, '▼', fonts['lg'], WHITE, FileBrowser._scroll_dn_r.centerx, FileBrowser._scroll_dn_r.centery, 'center')

        list_w  = panel.width - arrow_w - 14
        visible = self._visible_rows(row_h, list_h)
        y0      = list_top

        if self.current != self.current.root:
            up_r = pygame.Rect(panel.x + 6, y0, list_w - 6, row_h - 4)
            draw_rect(surf, DGREY, up_r, radius=4)
            draw_text(surf, '↑  ..', fonts['sm'], GREY, up_r.x + 12, up_r.centery, 'midleft')
            y0 += row_h
            visible -= 1

        for i, entry in enumerate(self.entries[self.scroll:self.scroll + visible]):
            er = pygame.Rect(panel.x + 6, y0 + i * row_h, list_w - 6, row_h - 4)
            draw_rect(surf, CARD if i % 2 == 0 else SURFACE, er, radius=4)
            icon = '/  ' if entry.is_dir() else '~  '
            tc   = AMBER if entry.is_dir() else WHITE
            draw_text(surf, icon + entry.name, fonts['sm'], tc, er.x + 12, er.centery, 'midleft')

        total = len(self.entries)
        vis   = self._visible_rows(row_h, list_h)
        if total > vis:
            draw_text(surf, f'{self.scroll+1}–{min(self.scroll+vis, total)} / {total}',
                      fonts['sm'], GREY, panel.centerx, panel.bottom - 8, 'midbottom')

    def _visible_rows(self, row_h, list_h):
        return max(1, list_h // row_h)

    def handle_event(self, event):
        if not self.visible:
            return None

        panel    = pygame.Rect(30, 30, W - 60, H - 60)
        btn_w    = max(60, int(panel.width * 0.08))
        row_h    = max(44, int(panel.height * 0.09))
        header_h = row_h + 4
        list_top = panel.y + header_h + row_h
        list_h   = panel.bottom - 8 - list_top
        list_w   = panel.width - btn_w - 14
        visible  = self._visible_rows(row_h, list_h)
        has_up   = self.current != Path(self.current.root)

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            if FileBrowser._close_r and FileBrowser._close_r.collidepoint(pos):
                self.visible = False
                return 'close'

            if FileBrowser._scroll_up_r and FileBrowser._scroll_up_r.collidepoint(pos):
                self.scroll = max(0, self.scroll - 1)
                return None
            if FileBrowser._scroll_dn_r and FileBrowser._scroll_dn_r.collidepoint(pos):
                self.scroll = min(max(0, len(self.entries) - visible), self.scroll + 1)
                return None

            y0 = list_top
            if has_up:
                up_r = pygame.Rect(panel.x + 6, y0, list_w - 6, row_h - 4)
                if up_r.collidepoint(pos):
                    self.current = self.current.parent
                    self._scan()
                    return None
                y0      += row_h
                visible -= 1

            for i, entry in enumerate(self.entries[self.scroll:self.scroll + visible]):
                er = pygame.Rect(panel.x + 6, y0 + i * row_h, list_w - 6, row_h - 4)
                if er.collidepoint(pos):
                    if entry.is_dir():
                        self.current = entry
                        self._scan()
                        return None
                    else:
                        self.visible = False
                        return entry

            if not panel.collidepoint(pos):
                self.visible = False
                return 'close'

        elif event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(self.scroll - event.y, max(0, len(self.entries) - visible)))

        return None


# ─── App Icon ─────────────────────────────────────────────────────────────────

def _make_axe_icon():
    """Draw a small axe icon for the window titlebar / taskbar."""
    s = pygame.Surface((32, 32), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))

    # Handle — warm brown diagonal, bottom-left → upper-centre-right
    h_col  = (160, 100, 40)
    h_dark = (110,  65, 20)
    pygame.draw.line(s, h_col,  (5, 29), (21, 9), 4)
    pygame.draw.line(s, h_dark, (6, 30), (22, 10), 1)   # shadow edge

    # Axe head — silver-grey polygon (blade faces lower-left)
    blade  = (175, 188, 198)
    bevel  = (215, 225, 232)
    back   = (120, 132, 140)
    # Main head body
    pygame.draw.polygon(s, blade, [(17, 5), (28, 10), (25, 22), (15, 14)])
    # Back of head (poll)
    pygame.draw.polygon(s, back,  [(17, 5), (21, 5), (22, 9), (19, 13), (15, 14)])
    # Cutting edge highlight
    pygame.draw.line(s, bevel, (25, 22), (28, 10), 2)

    return s


# ─── Main App ─────────────────────────────────────────────────────────────────

class WoodshedApp:
    def __init__(self):
        # sounddevice handles audio; prevent pygame from claiming the audio device
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        pygame.init()

        global W, H
        info = pygame.display.Info()
        W, H = info.current_w, info.current_h

        pygame.display.set_icon(_make_axe_icon())
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('WOODSHED')

        scale = H / 533
        global FONT_MONO, FONT_XS, FONT_SM, FONT_LG, FONT_XL
        try:
            FONT_XS   = pygame.font.SysFont('dejavusansmono', max(9,  int(10 * scale)))
            FONT_SM   = pygame.font.SysFont('dejavusansmono', max(12, int(14 * scale)))
            FONT_MONO = pygame.font.SysFont('dejavusansmono', max(13, int(15 * scale)))
            FONT_LG   = pygame.font.SysFont('dejavusansmono', max(16, int(20 * scale)), bold=True)
            FONT_XL   = pygame.font.SysFont('dejavusansmono', max(24, int(34 * scale)), bold=True)
        except Exception:
            FONT_XS   = pygame.font.SysFont('monospace', max(9,  int(10 * scale)))
            FONT_SM   = pygame.font.SysFont('monospace', max(12, int(14 * scale)))
            FONT_MONO = pygame.font.SysFont('monospace', max(13, int(15 * scale)))
            FONT_LG   = pygame.font.SysFont('monospace', max(16, int(20 * scale)), bold=True)
            FONT_XL   = pygame.font.SysFont('monospace', max(24, int(34 * scale)), bold=True)

        self.fonts  = {'xs': FONT_XS, 'sm': FONT_SM, 'mono': FONT_MONO, 'lg': FONT_LG, 'xl': FONT_XL}
        self.clock  = pygame.time.Clock()
        self.engine = AudioEngine()
        self.engine.init_stream()
        self.browser = FileBrowser()
        self._build_ui()
        self.status = 'Load a file to begin'
        self._marker_scroll = 0
        self._sect_scroll   = 0
        self._renaming_marker = None   # name of marker currently being renamed (None = not renaming)
        self._renaming_section   = None   # name of saved loop currently being renamed
        self._last_jump_pos = None     # raw sample index of last explicit seek / marker jump

    def _build_ui(self):  # noqa: C901
        pad = max(5, W // 170)
        g   = pad * 2

        # ── Waveform ──────────────────────────────────────────────────────────
        wh = int(H * 0.175)
        self.waveform = Waveform((pad, pad, W - 2*pad, wh))

        # ── Inline marker bar (just below waveform) ───────────────────────────
        mk_inline_h = max(32, int(H * 0.065))
        mk_top      = wh + pad * 2
        self.marker_rect = pygame.Rect(pad, mk_top, W - 2*pad, mk_inline_h)

        nbw  = max(50, int(W * 0.060))   # nav button width
        mbh  = mk_inline_h - 4           # button height inside bar
        mby  = mk_top + 2

        # Zoom/pan buttons sit on the waveform itself
        wf_r    = self.waveform.rect
        zbw_s   = max(22, int(W * 0.028))
        zbg     = 3
        zfit_w  = max(30, int(W * 0.038))
        zbh_s   = max(16, int(wh * 0.22))
        # Pan — top-left corner of waveform
        zbh_pan = max(16, int(wh * 0.25))
        self.btn_pan_left  = Button((wf_r.x + 2,               wf_r.y + 2, zbw_s, zbh_pan), '◀', color=DGREY)
        self.btn_pan_right = Button((wf_r.x + 2 + zbw_s + zbg, wf_r.y + 2, zbw_s, zbh_pan), '▶', color=DGREY)
        # Zoom — bottom-left corner of waveform
        zby = wf_r.bottom - zbh_s - 2
        self.btn_zoom_out = Button((wf_r.x + 2,                    zby, zbw_s,  zbh_s), '−',   color=DGREY)
        self.btn_zoom_in  = Button((wf_r.x + 2 + zbw_s + zbg,     zby, zbw_s,  zbh_s), '+',   color=DGREY)
        self.btn_zoom_fit = Button((wf_r.x + 2 + 2*(zbw_s + zbg), zby, zfit_w, zbh_s), 'FIT', color=DGREY)
        nav_gap = 3   # tight gap so the two nav buttons read as a pair
        self.btn_next_marker = Button((W - pad - nbw,               mby, nbw, mbh), 'MKR ▶', color=DGREY)
        self.btn_prev_marker = Button((W - pad - nbw - nav_gap - nbw, mby, nbw, mbh), '◀ MKR', color=DGREY)

        # + MKR button lives in the marker bar, just left of the nav pair
        mkr_add_w = max(60, int(nbw * 1.2))
        add_gap   = 8
        self.btn_mkr = Button((self.btn_prev_marker.rect.left - add_gap - mkr_add_w, mby, mkr_add_w, mbh), '+ MKR',
                              color=BLUE, text_color=BG)

        # Marker scroll arrows — left and right edges of the marker bar content area
        scroll_bw   = max(16, int(mbh * 0.55))   # narrower than square
        mkr_nav_gap = 10    # gap between right scroll button and +MKR button
        mr_r        = self.marker_rect
        self.btn_mkr_scroll_left  = Button((mr_r.x + 2,                                          mby, scroll_bw, mbh), '◀', color=DGREY)
        self.btn_mkr_scroll_right = Button((self.btn_mkr.rect.left - scroll_bw - mkr_nav_gap,    mby, scroll_bw, mbh), '▶', color=DGREY)

        # ── Saved loop bar (just below marker bar) ────────────────────────────
        lp_top      = mk_top + mk_inline_h + g
        lp_inline_h = mk_inline_h   # same height as marker bar
        self.sect_bar_rect = pygame.Rect(pad, lp_top, W - 2*pad, lp_inline_h)

        lp_bh = lp_inline_h - 4
        lp_by = lp_top + 2

        lp_add_w   = max(60, int(nbw * 1.2))
        lp_add_gap = 8
        self.btn_sect = Button((W - pad - lp_add_w, lp_by, lp_add_w, lp_bh), '+ SECT',
                             color=PURPLE, text_color=BG)

        lp_scroll_bw  = max(16, int(lp_bh * 0.55))
        lp_scroll_gap = 10
        lr_r = self.sect_bar_rect
        self.btn_sect_scroll_left  = Button((lr_r.x + 2,                                               lp_by, lp_scroll_bw, lp_bh), '◀', color=DGREY)
        self.btn_sect_scroll_right = Button((self.btn_sect.rect.left - lp_scroll_bw - lp_scroll_gap,     lp_by, lp_scroll_bw, lp_bh), '▶', color=DGREY)

        # ── Row 1: File / session ops (starts after loop bar) ─────────────────
        # Both rows use explicit gaps; last button always reaches W-pad exactly.
        bh1 = int(H * 0.058)
        by1 = lp_top + lp_inline_h + g
        n1  = 3
        bw1 = (W - 2*pad - (n1 - 1) * g) // n1
        bx1 = pad
        r1  = []
        for i in range(n1):
            w = bw1 if i < n1 - 1 else (W - pad - bx1)
            r1.append((bx1, by1, w, bh1))
            bx1 += w + g
        self.btn_open = Button(r1[0], 'OPEN')
        self.btn_save = Button(r1[1], 'SAVE')
        self.btn_load = Button(r1[2], 'LOAD')

        # ── Row 2: Transport + Loop controls ─────────────────────────────────
        bh2 = int(H * 0.105)
        by2 = by1 + bh1 + g

        # ◀◀  PLAY  ■  ▶▶  REPLAY  |  LOOP  [A  +A  B]  +B  CLR
        pct2 = [8, 13, 8, 8, 9, 9, 6, 6, 6, 6, 9]
        n2   = len(pct2)
        avail2 = W - 2*pad - (n2 - 1) * g
        unit2  = avail2 / sum(pct2)
        bx = pad
        tr = []
        for i, p in enumerate(pct2):
            w = int(p * unit2) if i < n2 - 1 else (W - pad - bx)
            tr.append((bx, by2, w, bh2))
            bx += w + g

        self.btn_seek_back = Button(tr[0],  '◀◀',     color=DGREY)
        self.btn_play      = Button(tr[1],  '▶  PLAY', color=GREEN, text_color=BG)
        self.btn_stop      = Button(tr[2],  '■',       color=DGREY)
        self.btn_seek_fwd  = Button(tr[3],  '▶▶',     color=DGREY)
        self.btn_replay    = Button(tr[4],  'REPLAY',  toggle=True, active=True)
        self.btn_loop      = Button(tr[5],  'LOOP',    toggle=True, active_color=PURPLE)
        self.btn_a         = Button(tr[6],  '[ A',     color=DGREY, text_color=GREEN)
        self.btn_mkr_a     = Button(tr[7],  '+ A',     color=DGREY, text_color=GREEN)
        self.btn_b         = Button(tr[8],  'B ]',     color=DGREY, text_color=RED)
        self.btn_mkr_b     = Button(tr[9],  '+ B',     color=DGREY, text_color=RED)
        self.btn_clrloop   = Button(tr[10], 'CLR AB')

        # ── Sliders — 3 columns ───────────────────────────────────────────────
        sl_h    = max(10, int(H * 0.022))
        thumb_r = max(8, sl_h + 4)
        col_m   = pad * 6
        sw      = (W - 2*pad - 2*col_m) // 3
        sx      = [pad + i*(sw + col_m) for i in range(3)]

        lbl_gap = thumb_r + 4 + int(H * 0.068)
        row_gap = sl_h + pad * 2 + lbl_gap

        sy1 = by2 + bh2 + pad + lbl_gap
        sy2 = sy1 + row_gap
        sy3 = sy2 + row_gap

        self.sl_speed = Slider((sx[0], sy1-20, sw, sl_h), 0.25, 2.0, 1.0, 'SPEED',
                               fmt='{:.0%}', color=GREEN, step=0.05, default=1.0, toggleable=True)
        self.sl_pitch = Slider((sx[1], sy1-20, sw, sl_h), -12,  12,  0,   'PITCH',
                               fmt='{:+.0f} st', color=GREEN, step=1, default=0, toggleable=True)
        self.sl_vol   = Slider((sx[2], sy1-20, sw, sl_h), 0.0,  1.0, 0.8, 'VOLUME',
                               fmt='{:.0%}', color=GREEN, step=0.05, default=0.8, toggleable=True)

        self.sl_eq_lo = Slider((sx[0], sy2-40, sw, sl_h), -12, 12, 0, 'EQ LOW',
                               fmt='{:+.1f}dB', color=AMBER, step=1.0, default=0, toggleable=True)
        self.sl_eq_md = Slider((sx[1], sy2-40, sw, sl_h), -12, 12, 0, 'EQ MID',
                               fmt='{:+.1f}dB', color=AMBER, step=1.0, default=0, toggleable=True)
        self.sl_eq_hi = Slider((sx[2], sy2-40, sw, sl_h), -12, 12, 0, 'EQ HIGH',
                               fmt='{:+.1f}dB', color=AMBER, step=1.0, default=0, toggleable=True)

        sw4  = (W - 2*pad - 3*col_m) // 4
        sx4  = [pad + i*(sw4 + col_m) for i in range(4)]
        self.sl_tr_start  = Slider((sx4[0], sy3-55, sw4, sl_h), 0.25, 1.0, 0.5,  'TRAIN START',
                                   fmt='{:.0%}', color=RED, step=0.05, default=0.5)
        self.sl_tr_target = Slider((sx4[1], sy3-55, sw4, sl_h), 0.25, 1.2, 1.0,  'TRAIN TARGET',
                                   fmt='{:.0%}', color=RED, step=0.05, default=1.0)
        self.sl_tr_step   = Slider((sx4[2], sy3-55, sw4, sl_h), 0.01, 0.20, 0.05,'STEP',
                                   fmt='{:.0%}', color=RED, step=0.01, default=0.05)
        self.sl_tr_reps   = Slider((sx4[3], sy3-55, sw4, sl_h), 1,   10,   2,    'REPS/STEP',
                                   fmt='{:.0f}x', color=RED, step=1,    default=2)

        self._eq_toggle_rect      = None   # set each frame in _draw; used for click detection
        self._trainer_toggle_rect = None   # set each frame in _draw; used for click detection
        self._trainer_curve_rect  = None   # CURVE mode toggle

        self._sy_labels = {
            'playback': sy1 - lbl_gap,
            'eq':       sy2 - lbl_gap,
            'trainer':  sy3 - lbl_gap,
        }

        status_h       = int(H * 0.068)
        self.info_rect = pygame.Rect(pad, H - status_h - pad - 50, W - 2*pad, status_h)

        self.all_sliders = [
            self.sl_speed, self.sl_pitch, self.sl_vol,
            self.sl_eq_lo, self.sl_eq_md, self.sl_eq_hi,
            self.sl_tr_start, self.sl_tr_target, self.sl_tr_step, self.sl_tr_reps,
        ]
        self.all_buttons = [
            self.btn_open, self.btn_save, self.btn_load, self.btn_mkr,
            self.btn_seek_back, self.btn_play, self.btn_stop, self.btn_seek_fwd,
            self.btn_replay,
            self.btn_loop, self.btn_a, self.btn_mkr_a, self.btn_b, self.btn_mkr_b,
            self.btn_clrloop,
            self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_fit,
            self.btn_pan_left, self.btn_pan_right,
            self.btn_prev_marker, self.btn_next_marker,
            self.btn_mkr_scroll_left, self.btn_mkr_scroll_right,
            self.btn_sect, self.btn_sect_scroll_left, self.btn_sect_scroll_right,
        ]

    def _sync_sliders_to_engine(self):
        e = self.engine
        if e.trainer_on:
            # Trainer owns speed — reflect its current value in the slider
            self.sl_speed.value = e._trainer_current
            # Allow live adjustment of reps while training
            e.trainer_reps = int(self.sl_tr_reps.value)
        else:
            e.speed = self.sl_speed.value if self.sl_speed.enabled else 1.0
        e.pitch   = round(self.sl_pitch.value) if self.sl_pitch.enabled else 0
        e.volume  = self.sl_vol.value   if self.sl_vol.enabled   else 1.0
        e.eq_low  = self.sl_eq_lo.value if self.sl_eq_lo.enabled else 0.0
        e.eq_mid  = self.sl_eq_md.value if self.sl_eq_md.enabled else 0.0
        e.eq_high = self.sl_eq_hi.value if self.sl_eq_hi.enabled else 0.0

    def run(self):
        running = True
        marker_input_mode = False
        marker_name_buf   = ''
        sect_input_mode   = False
        sect_name_buf     = ''

        while running:
            target_fps = FPS if self.engine._playing else 20
            self.clock.tick(target_fps)

            # Sync slider values to engine every frame — chunk loop picks up new values
            self._sync_sliders_to_engine()

            # Auto-scroll waveform to keep playhead in view when playing and zoomed
            if self.engine._playing:
                self.waveform.follow_playhead(self.engine.position_fraction)

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if marker_input_mode:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            old = self._renaming_marker
                            if old is not None and old in self.engine.markers:
                                new_name = marker_name_buf.strip()
                                if new_name:
                                    pos = self.engine.markers.pop(old)
                                    self.engine.markers[new_name] = pos
                                    self.status = f'Marker renamed → "{new_name}"'
                                else:
                                    del self.engine.markers[old]
                                    self.status = f'Marker "{old}" deleted'
                            marker_input_mode = False
                            marker_name_buf   = ''
                            self._renaming_marker = None
                        elif event.key == pygame.K_ESCAPE:
                            marker_input_mode = False
                            marker_name_buf   = ''
                            self._renaming_marker = None
                        elif event.key == pygame.K_BACKSPACE:
                            marker_name_buf = marker_name_buf[:-1]
                        else:
                            marker_name_buf += event.unicode
                    continue

                if sect_input_mode:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            old = self._renaming_section
                            if old is not None and old in self.engine.saved_sections:
                                new_name = sect_name_buf.strip()
                                if new_name:
                                    val = self.engine.saved_sections.pop(old)
                                    self.engine.saved_sections[new_name] = val
                                    self.status = f'Section renamed → "{new_name}"'
                                else:
                                    del self.engine.saved_sections[old]
                                    self.status = f'Section "{old}" deleted'
                            sect_input_mode   = False
                            sect_name_buf     = ''
                            self._renaming_section = None
                        elif event.key == pygame.K_ESCAPE:
                            sect_input_mode   = False
                            sect_name_buf     = ''
                            self._renaming_section = None
                        elif event.key == pygame.K_BACKSPACE:
                            sect_name_buf = sect_name_buf[:-1]
                        else:
                            sect_name_buf += event.unicode
                    continue

                if self.browser.visible:
                    result = self.browser.handle_event(event)
                    if result and result != 'close':
                        chosen = result
                        self.status = f'Loading {chosen.name}…'
                        self.browser.visible = False
                        def _bg_load(path=chosen):
                            try:
                                self.engine.load(path)
                                self.waveform.build(self.engine.raw)
                                self.status = f'Loaded: {path.name}'
                            except Exception as ex:
                                self.status = f'Error: {ex}'
                        threading.Thread(target=_bg_load, daemon=True).start()
                    continue

                # Process buttons first so waveform drag doesn't fire under zoom buttons
                btn_consumed = False
                for btn in self.all_buttons:
                    if btn.handle_event(event):
                        self._on_button(btn, marker_input_mode)
                        if btn is self.btn_mkr:
                            if self.engine.raw is not None:
                                self._add_marker(self.engine._actual_pos)
                        btn_consumed = True

                if not btn_consumed:
                    wf_result = self.waveform.handle_event(
                        event,
                        extra_rects=(self.btn_pan_left.rect, self.btn_pan_right.rect),
                    )
                    if isinstance(wf_result, tuple):
                        # Drag-select → set loop region
                        a_frac, b_frac = wf_result
                        e = self.engine
                        if e.raw is not None:
                            e.loop_a = int(a_frac * len(e.raw))
                            e.loop_b = int(b_frac * len(e.raw))
                            e.loop_on = True
                            self.btn_loop.active = True
                            dur = e.duration_seconds
                            self.status = (f'Loop: {fmt_time(a_frac * dur)}'
                                           f' → {fmt_time(b_frac * dur)}')
                    elif wf_result is not None:
                        self.engine.seek(wf_result)
                        if self.engine.raw is not None:
                            self._last_jump_pos = int(wf_result * len(self.engine.raw))

                for sl in self.all_sliders:
                    sl.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._eq_toggle_rect and self._eq_toggle_rect.collidepoint(event.pos):
                        self.engine.eq_enabled = not self.engine.eq_enabled
                    elif self._trainer_curve_rect and self._trainer_curve_rect.collidepoint(event.pos):
                        self.engine.trainer_nonlinear = not self.engine.trainer_nonlinear
                    elif self._trainer_toggle_rect and self._trainer_toggle_rect.collidepoint(event.pos):
                        e = self.engine
                        if not e.trainer_on:
                            if e.raw is None:
                                self.status = 'Load a file first'
                            else:
                                e.trainer_start  = self.sl_tr_start.value
                                e.trainer_target = self.sl_tr_target.value
                                e.trainer_step   = self.sl_tr_step.value
                                e.trainer_reps   = int(self.sl_tr_reps.value)
                                e.start_trainer()
                                self.sl_speed.value = e.speed
                                if not e._playing:
                                    e.play()
                                    self.btn_play.label = '|| PAUSE'
                                self.status = f'Trainer: {e.trainer_start:.0%} → {e.trainer_target:.0%}'
                        else:
                            e.stop_trainer()
                            self.status = 'Trainer stopped'

                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    mx, my = event.pos
                    mr  = self.marker_rect
                    if mr.collidepoint(mx, my) and self.engine.markers:
                        sorted_markers = sorted(self.engine.markers.items(), key=lambda kv: kv[1])
                        content_x = self.btn_mkr_scroll_left.rect.right + 8
                        content_r = self.btn_mkr_scroll_right.rect.left - 8
                        avail     = max(1, content_r - content_x)
                        btn_h     = mr.height - 4
                        btn_y     = mr.y + 2
                        _mkr_gap  = 2
                        PEEK      = 28
                        base_w    = max(40, int(avail / 6.6))
                        widths    = [max(base_w, self.fonts['sm'].size(n)[0] + 20)
                                     for n, _ in sorted_markers]
                        full_avail = max(1, avail - 2 * PEEK)
                        visible = 0
                        used    = 0
                        for w in widths[self._marker_scroll:]:
                            if used + w > full_avail:
                                break
                            used += w + _mkr_gap
                            visible += 1
                        visible = max(1, visible)
                        scroll   = max(0, min(self._marker_scroll, max(0, len(sorted_markers) - visible)))
                        has_left = scroll > 0
                        full_x   = content_x + (PEEK if has_left else 0)
                        x = full_x
                        for i in range(scroll, scroll + visible):
                            name, idx = sorted_markers[i]
                            w  = widths[i]
                            br = pygame.Rect(x, btn_y, w - _mkr_gap, btn_h)
                            if br.collidepoint(mx, my):
                                if event.button == 3:
                                    self._renaming_marker = name
                                    marker_input_mode = True
                                    marker_name_buf   = name
                                else:
                                    self.engine.goto_marker(name)
                                    self._last_jump_pos = idx
                                    self.status = f'→ marker "{name}"'
                            x += w + _mkr_gap

                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    mx, my = event.pos
                    lr = self.sect_bar_rect
                    if lr.collidepoint(mx, my) and self.engine.saved_sections:
                        sorted_sections  = sorted(self.engine.saved_sections.items(), key=lambda kv: kv[1][0])
                        content_x = self.btn_sect_scroll_left.rect.right + 8
                        content_r = self.btn_sect_scroll_right.rect.left - 8
                        avail     = max(1, content_r - content_x)
                        btn_h     = lr.height - 4
                        btn_y     = lr.y + 2
                        _lp_gap   = 2
                        PEEK      = 28
                        base_w    = max(40, int(avail / 6.6))
                        widths    = [max(base_w, self.fonts['sm'].size(n)[0] + 20)
                                     for n, _ in sorted_sections]
                        full_avail = max(1, avail - 2 * PEEK)
                        visible = 0
                        used    = 0
                        for w in widths[self._sect_scroll:]:
                            if used + w > full_avail:
                                break
                            used += w + _lp_gap
                            visible += 1
                        visible  = max(1, visible)
                        scroll   = max(0, min(self._sect_scroll, max(0, len(sorted_sections) - visible)))
                        has_left = scroll > 0
                        full_x   = content_x + (PEEK if has_left else 0)
                        x = full_x
                        for i in range(scroll, scroll + visible):
                            name, (la, lb) = sorted_sections[i]
                            w  = widths[i]
                            br = pygame.Rect(x, btn_y, w - _lp_gap, btn_h)
                            if br.collidepoint(mx, my):
                                if event.button == 3:
                                    self._renaming_section = name
                                    sect_input_mode = True
                                    sect_name_buf   = name
                                else:
                                    e = self.engine
                                    e.loop_a = la
                                    e.loop_b = lb
                                    e.loop_on = True
                                    self.btn_loop.active = True
                                    if e.raw is not None:
                                        dur = e.duration_seconds
                                        self.status = (f'Section "{name}": '
                                                       f'{fmt_time(la/e.sr)} → {fmt_time(lb/e.sr)}')
                            x += w + _lp_gap

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._toggle_play()
                    elif event.key == pygame.K_LEFT:
                        self.engine.seek_seconds(self.engine.position_seconds - 5)
                    elif event.key == pygame.K_RIGHT:
                        self.engine.seek_seconds(self.engine.position_seconds + 5)
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # ── Draw ──────────────────────────────────────────────────────────
            self.screen.fill(BG)
            self._draw(marker_input_mode, marker_name_buf, sect_input_mode, sect_name_buf)
            pygame.display.flip()

        pygame.quit()

    def _auto_marker_name(self):
        n = 1
        while str(n) in self.engine.markers:
            n += 1
        return str(n)

    def _auto_section_name(self):
        n = 1
        while str(n) in self.engine.saved_sections:
            n += 1
        return str(n)

    def _add_marker(self, raw_pos):
        name = self._auto_marker_name()
        self.engine.markers[name] = raw_pos
        self.status = f'Marker "{name}" added'
        return name

    def _on_button(self, btn, marker_mode):
        e = self.engine
        if btn is self.btn_open:
            self.browser.visible = True
        elif btn is self.btn_play:
            self._toggle_play()
        elif btn is self.btn_stop:
            e.pause()
            e.seek(0)
            self.btn_play.label = '▶  PLAY'
        elif btn is self.btn_seek_back:
            e.seek_seconds(e.position_seconds - 5)
        elif btn is self.btn_seek_fwd:
            e.seek_seconds(e.position_seconds + 5)
        elif btn is self.btn_loop:
            e.loop_on = btn.active
        elif btn is self.btn_a:
            e.set_loop_a()
            self.status = f'Loop A set @ {fmt_time(e.position_seconds)}'
        elif btn is self.btn_mkr_a:
            if e.loop_a is not None:
                self._add_marker(e.loop_a)
            else:
                self.status = 'Set loop A first'
        elif btn is self.btn_b:
            e.set_loop_b()
            self.status = f'Loop B set @ {fmt_time(e.position_seconds)}'
        elif btn is self.btn_mkr_b:
            if e.loop_b is not None:
                self._add_marker(e.loop_b)
            else:
                self.status = 'Set loop B first'
        elif btn is self.btn_clrloop:
            e.clear_loop()
            self.btn_loop.active = False
            self.status = 'Loop cleared'
        elif btn is self.btn_save:
            p = Path.home() / 'woodshed_session.json'
            e.save_session(p)
            self.status = f'Saved → {p.name}'
        elif btn is self.btn_load:
            p = Path.home() / 'woodshed_session.json'
            if p.exists():
                self.status = 'Loading session…'
                def _bg_load_session():
                    try:
                        e.load_session(p)
                        self.waveform.build(e.raw)
                        self.sl_speed.value = e.speed
                        self.sl_pitch.value = e.pitch
                        self.sl_vol.value   = e.volume
                        self.status = 'Session loaded'
                    except Exception as ex:
                        self.status = f'Error: {ex}'
                threading.Thread(target=_bg_load_session, daemon=True).start()
            else:
                self.status = 'No saved session found'
        elif btn is self.btn_zoom_in:
            self.waveform.zoom_in()
        elif btn is self.btn_zoom_out:
            self.waveform.zoom_out()
        elif btn is self.btn_pan_left:
            self.waveform.pan(-0.25 / self.waveform.zoom)
        elif btn is self.btn_pan_right:
            self.waveform.pan(0.25 / self.waveform.zoom)
        elif btn is self.btn_zoom_fit:
            if e.loop_a is not None and e.loop_b is not None and e.raw is not None:
                total = len(e.raw)
                self.waveform.zoom_to_fit(e.loop_a / total, e.loop_b / total)
            else:
                self.waveform.zoom = 1.0
                self.waveform.view_start = 0.0
        elif btn is self.btn_mkr_scroll_left:
            self._marker_scroll = max(0, self._marker_scroll - 1)
        elif btn is self.btn_mkr_scroll_right:
            self._marker_scroll += 1   # clamped to valid range in _draw
        elif btn is self.btn_sect:
            if e.loop_a is not None and e.loop_b is not None:
                name = self._auto_section_name()
                e.saved_sections[name] = (e.loop_a, e.loop_b)
                self.status = f'Section "{name}" saved'
            else:
                self.status = 'Set loop A and B first'
        elif btn is self.btn_sect_scroll_left:
            self._sect_scroll = max(0, self._sect_scroll - 1)
        elif btn is self.btn_sect_scroll_right:
            self._sect_scroll += 1   # clamped to valid range in _draw
        elif btn is self.btn_prev_marker:
            e.goto_prev_marker()
            self._last_jump_pos = e._raw_pos
            self.status = '→ prev marker'
        elif btn is self.btn_next_marker:
            e.goto_next_marker()
            self._last_jump_pos = e._raw_pos
            self.status = '→ next marker'

    def _toggle_play(self):
        e = self.engine
        if e.raw is None:
            self.status = 'Load a file first'
            return
        if e._playing:
            e.pause()
            self.btn_play.label = '▶  PLAY'
        else:
            if self.btn_replay.active and self._last_jump_pos is not None:
                e.seek(self._last_jump_pos / len(e.raw))
            e.play()
            self.btn_play.label = '|| PAUSE'

    def _draw(self, marker_input_mode, marker_name_buf, sect_input_mode=False, sect_name_buf=''):
        e = self.engine
        f = self.fonts

        dur = e.duration_seconds
        la_frac = (e.loop_a / e.sr / dur) if (e.loop_a is not None and dur) else None
        lb_frac = (e.loop_b / e.sr / dur) if (e.loop_b is not None and dur) else None
        jump_frac = (self._last_jump_pos / len(e.raw)) if (
            self._last_jump_pos is not None and e.raw is not None
        ) else None
        self.waveform.draw(self.screen, e.position_fraction, la_frac, lb_frac,
                           e.markers, dur, jump_frac=jump_frac, loop_active=e.loop_on,
                           saved_sections=e.saved_sections, sr=e.sr)

        # Zoom level indicator — to the right of the FIT button, inside waveform bottom
        if self.waveform.zoom > 1.0:
            ind_x = self.btn_zoom_fit.rect.right + 6
            ind_y = self.btn_zoom_fit.rect.centery
            draw_text(self.screen, f'{self.waveform.zoom:.0f}×', f['sm'], PURPLE_DIM, ind_x, ind_y, 'midleft')

        wbottom = self.waveform.rect.bottom
        draw_text(self.screen, fmt_time(e.position_seconds), f['lg'], PURPLE_DIM, W//2, wbottom - 6, 'midbottom')
        draw_text(self.screen, '/ ' + fmt_time(dur),
                  f['sm'], GREY, W//2 + int(W*0.10), wbottom - 6, 'midbottom')

        if e.trainer_on:
            if e.trainer_nonlinear and e.trainer_start > 0:
                _ratio    = e._trainer_current / e.trainer_start
                _eff_reps = max(1, round(e.trainer_reps * _ratio * _ratio))
            else:
                _eff_reps = e.trainer_reps
            _rep_str = f'{e._trainer_rep_count + 1}/{_eff_reps}'
            _tr_str  = f'TRAINER  {e._trainer_current:.0%}  ·  {_rep_str}'
            wf_r = self.waveform.rect
            draw_text(self.screen, _tr_str, f['sm'], RED, wf_r.right - 6, wbottom - 6, 'bottomright')

        _bar_btns = (self.btn_mkr_scroll_left, self.btn_mkr_scroll_right,
                     self.btn_prev_marker,    self.btn_next_marker,
                     self.btn_mkr,
                     self.btn_sect_scroll_left, self.btn_sect_scroll_right,
                     self.btn_sect)
        for btn in self.all_buttons:
            if btn in _bar_btns:
                continue   # drawn after their bar backgrounds below
            btn.draw(self.screen, f)


        for sl in self.all_sliders:
            sl.draw(self.screen, f)

        lbl = self._sy_labels
        draw_text(self.screen, '─ PLAYBACK ───────────────────────────────────────────────────────────────────────────────────────────────',      f['sm'], BORDER, 10, lbl['playback']+5, 'topleft')
        # EQ section title — dot acts as master on/off toggle (same style as per-band labels)
        eq_lbl_y  = lbl['eq'] - 15
        dot_col   = GREEN if e.eq_enabled else GREY
        lbl_col   = WHITE if e.eq_enabled else GREY
        dot_cx    = 16
        dot_cy    = eq_lbl_y + f['sm'].get_height() // 2
        pygame.draw.circle(self.screen, dot_col, (dot_cx, dot_cy), 4)
        eq_tr = draw_text(self.screen, 'EQ', f['sm'], lbl_col, dot_cx + 10, eq_lbl_y, 'topleft')
        self._eq_toggle_rect = pygame.Rect(dot_cx - 6, eq_lbl_y - 2, eq_tr.right - dot_cx + 12, eq_tr.height + 4)
        draw_text(self.screen, '─────────────────────────────────────────────────────────────────────────────────────────────────────', f['sm'], BORDER, eq_tr.right + 6, eq_lbl_y, 'topleft')
        tr_lbl_y  = lbl['trainer'] - 30
        tr_dot_col = GREEN if e.trainer_on else GREY
        tr_lbl_col = WHITE if e.trainer_on else GREY
        tr_dot_cx  = 16
        tr_dot_cy  = tr_lbl_y + f['sm'].get_height() // 2
        pygame.draw.circle(self.screen, tr_dot_col, (tr_dot_cx, tr_dot_cy), 4)
        tr_tr = draw_text(self.screen, 'SPEED TRAINER', f['sm'], tr_lbl_col, tr_dot_cx + 10, tr_lbl_y, 'topleft')
        self._trainer_toggle_rect = pygame.Rect(tr_dot_cx - 6, tr_lbl_y - 2, tr_tr.right - tr_dot_cx + 12, tr_tr.height + 4)
        # CURVE toggle — non-linear reps mode
        cv_dot_x   = tr_tr.right + 24
        cv_dot_col = AMBER if e.trainer_nonlinear else GREY
        cv_lbl_col = WHITE if e.trainer_nonlinear else GREY
        pygame.draw.circle(self.screen, cv_dot_col, (cv_dot_x, tr_dot_cy), 4)
        cv_tr = draw_text(self.screen, 'CURVE', f['sm'], cv_lbl_col, cv_dot_x + 10, tr_lbl_y, 'topleft')
        self._trainer_curve_rect = pygame.Rect(cv_dot_x - 6, tr_lbl_y - 2, cv_tr.right - cv_dot_x + 12, cv_tr.height + 4)
        draw_text(self.screen, '─────────────────────────────────────────────────────────────────────────────────────────────────────', f['sm'], BORDER, cv_tr.right + 6, tr_lbl_y, 'topleft')

        # ── Inline marker bar (just below waveform) ───────────────────────────
        mr  = self.marker_rect
        draw_rect(self.screen, CARD, mr, radius=4)
        self.btn_mkr.draw(self.screen, f)
        self.btn_prev_marker.draw(self.screen, f)
        self.btn_next_marker.draw(self.screen, f)
        self.btn_mkr_scroll_left.draw(self.screen, f)
        self.btn_mkr_scroll_right.draw(self.screen, f)

        # Content area sits between the two scroll buttons with consistent padding
        _sc_pad  = 8   # gap between scroll button edge and marker content
        _mkr_gap = 2   # gap between individual marker buttons
        content_x = self.btn_mkr_scroll_left.rect.right  + _sc_pad
        content_r = self.btn_mkr_scroll_right.rect.left  - _sc_pad
        avail     = max(1, content_r - content_x)
        btn_h     = mr.height - 4
        btn_y     = mr.y + 2

        # Sorted markers (by waveform position, not insertion order)
        sorted_markers = sorted(e.markers.items(), key=lambda kv: kv[1]) if e.markers else []

        _scroll_off = (130, 130, 140)   # visible-but-muted: clearly legible, clearly not active

        if sorted_markers:
            _h_pad   = 10
            _mkr_gap = 2
            PEEK     = 28   # px shown on each peeking side

            base_w = max(40, int(avail / 6.6))
            # Per-button widths: each sized to its own content
            widths = [max(base_w, f['sm'].size(name)[0] + _h_pad * 2)
                      for name, _ in sorted_markers]

            # Count full buttons that fit, reserving PEEK on both sides
            full_avail = max(1, avail - 2 * PEEK)
            visible = 0
            used = 0
            for w in widths[self._marker_scroll:]:
                if used + w > full_avail:
                    break
                used += w + _mkr_gap
                visible += 1
            visible = max(1, visible)

            scroll = max(0, min(self._marker_scroll, max(0, len(sorted_markers) - visible)))
            self._marker_scroll = scroll
            has_left  = scroll > 0
            has_right = scroll + visible < len(sorted_markers)

            # x where the first fully-visible button starts (always left-aligned)
            full_x = content_x + (PEEK if has_left else 0)

            old_clip = self.screen.get_clip()
            self.screen.set_clip(pygame.Rect(content_x, mr.y, content_r - content_x, mr.height))

            # Left-peek button (no text — it's intentionally half-hidden)
            if has_left:
                px = full_x - widths[scroll - 1] - _mkr_gap
                draw_rect(self.screen, DGREY,
                          pygame.Rect(px, btn_y, widths[scroll - 1] - _mkr_gap, btn_h), radius=3)

            # Full buttons
            x = full_x
            for i in range(scroll, scroll + visible):
                name, idx = sorted_markers[i]
                w  = widths[i]
                t  = idx / e.sr
                br = pygame.Rect(x, btn_y, w - _mkr_gap, btn_h)
                draw_rect(self.screen, DGREY, br, radius=3)
                draw_text(self.screen, name,        f['sm'], BLUE,  br.centerx, br.y + 2,       'midtop')
                draw_text(self.screen, fmt_time(t), f['xs'], WHITE, br.centerx, br.bottom - 2, 'midbottom')
                x += w + _mkr_gap

            # Right-peek button (no text)
            if has_right:
                draw_rect(self.screen, DGREY,
                          pygame.Rect(x, btn_y, widths[scroll + visible] - _mkr_gap, btn_h), radius=3)

            self.screen.set_clip(old_clip)

            self.btn_mkr_scroll_left.text_color  = WHITE if has_left  else _scroll_off
            self.btn_mkr_scroll_right.text_color = WHITE if has_right else _scroll_off
        else:
            draw_text(self.screen, 'tap + MKR to add a marker', f['sm'], GREY,
                      content_x + avail // 2, mr.centery, 'center')
            self.btn_mkr_scroll_left.text_color  = _scroll_off
            self.btn_mkr_scroll_right.text_color = _scroll_off

        # ── Saved loop bar ────────────────────────────────────────────────────
        lr  = self.sect_bar_rect
        draw_rect(self.screen, CARD, lr, radius=4)
        self.btn_sect.draw(self.screen, f)
        self.btn_sect_scroll_left.draw(self.screen, f)
        self.btn_sect_scroll_right.draw(self.screen, f)

        _lp_sc_pad = 8
        _lp_gap    = 2
        lp_content_x = self.btn_sect_scroll_left.rect.right  + _lp_sc_pad
        lp_content_r = self.btn_sect_scroll_right.rect.left  - _lp_sc_pad
        lp_avail     = max(1, lp_content_r - lp_content_x)
        lp_btn_h     = lr.height - 4
        lp_btn_y     = lr.y + 2

        sorted_sections = sorted(e.saved_sections.items(), key=lambda kv: kv[1][0]) if e.saved_sections else []
        _lp_scroll_off = (130, 130, 140)

        if sorted_sections:
            _lp_h_pad = 10
            PEEK      = 28
            lp_base_w = max(40, int(lp_avail / 6.6))
            lp_widths = [max(lp_base_w, f['sm'].size(n)[0] + _lp_h_pad * 2)
                         for n, _ in sorted_sections]
            lp_full_avail = max(1, lp_avail - 2 * PEEK)
            lp_visible = 0
            lp_used    = 0
            for w in lp_widths[self._sect_scroll:]:
                if lp_used + w > lp_full_avail:
                    break
                lp_used += w + _lp_gap
                lp_visible += 1
            lp_visible = max(1, lp_visible)

            lp_scroll   = max(0, min(self._sect_scroll, max(0, len(sorted_sections) - lp_visible)))
            self._sect_scroll = lp_scroll
            lp_has_left  = lp_scroll > 0
            lp_has_right = lp_scroll + lp_visible < len(sorted_sections)
            lp_full_x    = lp_content_x + (PEEK if lp_has_left else 0)

            old_clip = self.screen.get_clip()
            self.screen.set_clip(pygame.Rect(lp_content_x, lr.y, lp_content_r - lp_content_x, lr.height))

            if lp_has_left:
                px = lp_full_x - lp_widths[lp_scroll - 1] - _lp_gap
                draw_rect(self.screen, DGREY,
                          pygame.Rect(px, lp_btn_y, lp_widths[lp_scroll - 1] - _lp_gap, lp_btn_h), radius=3)

            x = lp_full_x
            for i in range(lp_scroll, lp_scroll + lp_visible):
                name, (la, lb) = sorted_sections[i]
                w   = lp_widths[i]
                dur = (lb - la) / e.sr if e.sr else 0
                # Highlight if this is the active loop
                is_active = (e.loop_a == la and e.loop_b == lb)
                bg_col = PURPLE_DIM if is_active else DGREY
                br = pygame.Rect(x, lp_btn_y, w - _lp_gap, lp_btn_h)
                draw_rect(self.screen, bg_col, br, radius=3)
                draw_text(self.screen, name,            f['sm'], PURPLE, br.centerx, br.y + 2,       'midtop')
                draw_text(self.screen, fmt_time(dur),   f['xs'], WHITE,  br.centerx, br.bottom - 2, 'midbottom')
                x += w + _lp_gap

            if lp_has_right:
                draw_rect(self.screen, DGREY,
                          pygame.Rect(x, lp_btn_y, lp_widths[lp_scroll + lp_visible] - _lp_gap, lp_btn_h), radius=3)

            self.screen.set_clip(old_clip)

            self.btn_sect_scroll_left.text_color  = WHITE if lp_has_left  else _lp_scroll_off
            self.btn_sect_scroll_right.text_color = WHITE if lp_has_right else _lp_scroll_off
        else:
            draw_text(self.screen, 'tap + SECT to save current A/B as a section', f['sm'], GREY,
                      lp_content_x + lp_avail // 2, lr.centery, 'center')
            self.btn_sect_scroll_left.text_color  = _lp_scroll_off
            self.btn_sect_scroll_right.text_color = _lp_scroll_off

        if marker_input_mode:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 160))
            self.screen.blit(ov, (0, 0))
            box = pygame.Rect(W//2 - 280, H//2 - 70, 560, 140)
            draw_rect(self.screen, SURFACE, box, radius=8, border=2, border_color=AMBER)
            draw_text(self.screen, 'Rename marker:', f['lg'], AMBER, box.centerx, box.y + 14, 'midtop')
            draw_text(self.screen, marker_name_buf + '█', f['xl'], WHITE, box.centerx, box.y + 56, 'midtop')
            draw_text(self.screen, 'clear name + Enter to delete  ·  Esc to cancel', f['xs'], GREY, box.centerx, box.bottom - 8, 'midbottom')

        if sect_input_mode:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 160))
            self.screen.blit(ov, (0, 0))
            box = pygame.Rect(W//2 - 280, H//2 - 70, 560, 140)
            draw_rect(self.screen, SURFACE, box, radius=8, border=2, border_color=PURPLE)
            draw_text(self.screen, 'Rename section:', f['lg'], PURPLE, box.centerx, box.y + 14, 'midtop')
            draw_text(self.screen, sect_name_buf + '█', f['xl'], WHITE, box.centerx, box.y + 56, 'midtop')
            draw_text(self.screen, 'clear name + Enter to delete  ·  Esc to cancel', f['xs'], GREY, box.centerx, box.bottom - 8, 'midbottom')

        if self.browser.visible:
            self.browser.draw(self.screen, f)

        draw_rect(self.screen, SURFACE, self.info_rect, radius=3)
        fn   = Path(e.filepath).name if e.filepath else '—'
        info = f'{fn}  |  {self.status}'
        draw_text(self.screen, info, f['sm'], GREY, self.info_rect.x + 8, self.info_rect.centery, 'midleft')

        loop_txt = 'LOOP ON' if e.loop_on else 'LOOP OFF'
        loop_col = PURPLE if e.loop_on else (80, 50, 110)
        draw_text(self.screen, loop_txt, f['sm'], loop_col,
                  self.info_rect.right - 8, self.info_rect.centery, 'midright')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = WoodshedApp()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            app.engine.load(path)
            app.waveform.build(app.engine.raw)
            app.status = f'Loaded: {path.name}'
    app.run()
