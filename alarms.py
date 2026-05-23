"""
alarms.py — Audio alarm management for WakeMate v2.0.

Each alarm type is rate-limited to one fire per 5 seconds.
If the WAV files are missing or pygame fails, falls back to a system beep
so the application never crashes due to missing audio assets.
"""

from pygame import mixer
import time
import os

from app_logger import get_logger

_log = get_logger("alarms")

# ── Rate-limit state ─────────────────────────────────────────────────────────
last_alarm_times: dict[str, float] = {
    "eyes_closed": 0.0,
    "yawning":     0.0,
    "head_down":   0.0,
    "distracted":  0.0,
}


def can_sound_alarm(alarm_type: str) -> bool:
    """Return True (and update timestamp) if 5 s have passed since last alarm."""
    if time.time() - last_alarm_times[alarm_type] > 5.0:
        last_alarm_times[alarm_type] = time.time()
        return True
    return False


def _system_beep():
    """Cross-platform last-resort beep fallback."""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        print("\a", end="", flush=True)


def _ensure_mixer():
    """
    Fix 8 — Initialise pygame mixer if it has not been initialised yet.

    main.py and dashboard.py call mixer.init() at startup, but this guard
    makes alarms.py safe to use in any context (unit tests, standalone
    imports, etc.) without requiring the caller to have called mixer.init().
    """
    try:
        if not mixer.get_init():
            mixer.init()
    except Exception as e:
        _log.warning("Could not initialise pygame mixer: %s", e)


def _play_wav(filepath: str):
    """
    Attempt to load and play a WAV file via pygame.
    On any failure, emit a system beep instead of crashing.
    """
    _ensure_mixer()
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        mixer.music.load(filepath)
        mixer.music.play()
    except Exception as exc:
        _log.warning(
            "Audio playback failed for '%s': %s — falling back to system beep.",
            filepath, exc,
        )
        _system_beep()


# ── Public alarm functions ───────────────────────────────────────────────────

def sound_eyes_closed_alarm():
    if can_sound_alarm("eyes_closed"):
        _play_wav("sound.wav")


def sound_yawning_alarm():
    if can_sound_alarm("yawning"):
        _play_wav("music.wav")


def sound_head_down_alarm():
    if can_sound_alarm("head_down"):
        _play_wav("sound.wav")


def sound_distracted_alarm():
    if can_sound_alarm("distracted"):
        _play_wav("sound.wav")
