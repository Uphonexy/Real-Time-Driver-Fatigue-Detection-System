from pygame import mixer
import time

last_alarm_time = 0.0

def can_sound_alarm():
    global last_alarm_time
    if time.time() - last_alarm_time > 5.0:
        last_alarm_time = time.time()
        return True
    return False

def sound_eyes_closed_alarm():
    if can_sound_alarm():
        mixer.music.load("sound.wav")
        mixer.music.play()

def sound_yawning_alarm():
    if can_sound_alarm():
        mixer.music.load("music.wav")
        mixer.music.play()

def sound_head_down_alarm():
    if can_sound_alarm():
        # Fallback to sound.wav or another sound if available. We'll use sound.wav for head down instead of the yawning sound.
        mixer.music.load("sound.wav")
        mixer.music.play()

def sound_distracted_alarm():
    if can_sound_alarm():
        mixer.music.load("sound.wav")
        mixer.music.play()
