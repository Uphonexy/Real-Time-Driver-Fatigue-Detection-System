"""
alarms.py — Audio alarm management for WakeMate v2.0.
Rate-limited to one fire per 5 seconds per alarm type.
Falls back to a system beep if WAV files are missing or pygame fails.
"""
from pygame import mixer
import time, os

try:
    from app_logger import get_logger
    _log = get_logger("alarms")
except ImportError:
    import logging
    _log = logging.getLogger("alarms")

last_alarm_times: dict[str, float] = {
    "eyes_closed": 0.0,
    "yawning":     0.0,
    "head_down":   0.0,
    "distracted":  0.0,
}

def can_sound_alarm(alarm_type: str) -> bool:
    last_time = last_alarm_times.get(alarm_type, 0.0)
    if time.time() - last_time > 5.0:
        last_alarm_times[alarm_type] = time.time()
        return True
    return False

def _system_beep():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        print("\a", end="", flush=True)

def _ensure_mixer():
    try:
        if not mixer.get_init():
            mixer.init()
    except Exception as e:
        _log.warning("Could not initialise pygame mixer: %s", e)

def _play_wav(filepath: str):
    _ensure_mixer()
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        mixer.music.load(filepath)
        mixer.music.play()
    except Exception as exc:
        _log.warning("Audio playback failed for '%s': %s — using system beep.", filepath, exc)
        _system_beep()

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
