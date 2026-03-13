from pygame import mixer
import time

last_alarm_times = {
    "eyes_closed": 0.0,
    "yawning":     0.0,
    "head_down":   0.0,
    "distracted":  0.0
}

def can_sound_alarm(alarm_type):
    if time.time() - last_alarm_times[alarm_type] > 5.0:
        last_alarm_times[alarm_type] = time.time()
        return True
    return False

def sound_eyes_closed_alarm():
    if can_sound_alarm("eyes_closed"):
        mixer.music.load("sound.wav")
        mixer.music.play()

def sound_yawning_alarm():
    if can_sound_alarm("yawning"):
        mixer.music.load("music.wav")
        mixer.music.play()

def sound_head_down_alarm():
    if can_sound_alarm("head_down"):
        mixer.music.load("sound.wav")
        mixer.music.play()

def sound_distracted_alarm():
    if can_sound_alarm("distracted"):
        mixer.music.load("sound.wav")
        mixer.music.play()
