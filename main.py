import cv2
import dlib
import imutils
import time
import threading
from pygame import mixer
from imutils import face_utils

# Import separated modules
from detector import get_head_pose, eye_aspect_ratio, mouth_aspect_ratio, line_pairs
from thresholds import compute_adaptive_threshold
from calibration import run_calibration_phase
from alarms import (
    sound_eyes_closed_alarm, 
    sound_yawning_alarm, 
    sound_head_down_alarm, 
    sound_distracted_alarm
)
from logger import SessionLogger

# ──────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────
mixer.init()

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

# ──────────────────────────────────────────────
# Pre-Detection Input Screen
# ──────────────────────────────────────────────
age_group, drive_start_time = run_calibration_phase(cap)

# Threshold & System State Initialization
personal_baseline_ear = 0.25 # Initial fallback
personal_baseline_mar = 0.75
EYE_AR_THRESH = personal_baseline_ear 
MOU_AR_THRESH = personal_baseline_mar 

EYE_AR_CONSEC_FRAMES = 45   
HEAD_PITCH_THRESH = 10      
DISTRACTION_YAW_THRESH = 20

last_threshold_update = -1800 

COUNTER = 0
yawnStatus = False
yawns = 0

# ──────────────────────────────────────────────
# Session Logging & Calibration Initialization
# ──────────────────────────────────────────────
logger = SessionLogger(age_group)

calibration_mode = True
ear_samples = []
calibration_end_time = 0
calibration_duration = 300 # seconds

# ──────────────────────────────────────────────
# Main Detection Loop
# ──────────────────────────────────────────────
print("[INFO] Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read frame.")
        break

    frame = imutils.resize(frame, width=640)

    drive_minutes = (time.time() - drive_start_time) / 60
    hours = int(drive_minutes // 60)
    minutes = int(drive_minutes % 60)
    cv2.putText(frame, f"Drive Time: {hours:02d}:{minutes:02d}", (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Adaptive Threshold Update ──
    current_drive_seconds = time.time() - drive_start_time
    
    if calibration_mode:
        remaining_time = int(calibration_duration - current_drive_seconds)
        if remaining_time > 0:
            cv2.putText(frame, f"CALIBRATING - Drive normally", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Calibrating: {remaining_time // 60}:{remaining_time % 60:02d} remaining", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            calibration_mode = False
            calibration_end_time = time.time()
            if ear_samples:
                ear_samples.sort(reverse=True)
                top_70_percent_index = int(len(ear_samples) * 0.7)
                valid_samples = ear_samples[:top_70_percent_index]
                if valid_samples:
                    open_eye_avg = sum(valid_samples) / len(valid_samples)
                    personal_baseline_ear = max(0.20, open_eye_avg - 0.08)
                    logger.update_baseline(personal_baseline_ear)
                    
            print(f"[INFO] Calibration Complete. Open Eye Avg: {open_eye_avg:.3f}, Base Alert EAR: {personal_baseline_ear:.3f}")
            last_threshold_update = -1800
            
    if not calibration_mode and time.time() - calibration_end_time < 5:
        cv2.putText(frame, f"Calibration Complete - Baseline EAR: {personal_baseline_ear:.3f}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not calibration_mode and current_drive_seconds - last_threshold_update >= 1800:
        EYE_AR_THRESH = compute_adaptive_threshold(personal_baseline_ear, age_group, drive_minutes)
        MOU_AR_THRESH = compute_adaptive_threshold(personal_baseline_mar, age_group, drive_minutes)
        last_threshold_update = current_drive_seconds
        
    cv2.putText(frame, f"EAR Thresh: {EYE_AR_THRESH:.3f}", (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth    = shape[mStart:mEnd]

        leftEAR  = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear      = (leftEAR + rightEAR) / 2.0
        mouEAR   = mouth_aspect_ratio(mouth)
        
        if calibration_mode:
            ear_samples.append(ear)

        reprojectdst, euler_angle = get_head_pose(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        for start, end in line_pairs:
            start_point = tuple(map(int, reprojectdst[start]))
            end_point   = tuple(map(int, reprojectdst[end]))
            cv2.line(frame, start_point, end_point, (0, 0, 255))

        xx = euler_angle[0, 0]
        yy = euler_angle[1, 0]
        zz = euler_angle[2, 0]
        cv2.putText(frame, f"X: {xx:7.2f}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(frame, f"Y: {yy:7.2f}", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(frame, f"Z: {zz:7.2f}", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # ── Head Pose Check ──
        if xx > HEAD_PITCH_THRESH:
            cv2.putText(frame, "Head Down", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not calibration_mode:
                threading.Thread(name='sound_head_down_alarm', target=sound_head_down_alarm).start()
                logger.log_event("head_down", drive_minutes, xx, HEAD_PITCH_THRESH)

        # ── Distraction Check (Y Euler angle) ──
        if abs(yy) > DISTRACTION_YAW_THRESH:
            cv2.putText(frame, "DISTRACTED", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange
            if not calibration_mode:
                threading.Thread(name='sound_distracted_alarm', target=sound_distracted_alarm).start()
                logger.log_event("distracted", drive_minutes, yy, DISTRACTION_YAW_THRESH)

        # ── Eye Closure Check ──
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not calibration_mode:
                if COUNTER == 0: 
                    logger.log_event("eyes_closed", drive_minutes, ear, EYE_AR_THRESH)
                COUNTER += 1
                
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not calibration_mode:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                threading.Thread(name='sound_eyes_closed_alarm', target=sound_eyes_closed_alarm).start()
                if COUNTER == EYE_AR_CONSEC_FRAMES: 
                    logger.log_event("drowsiness_alarm", drive_minutes, ear, EYE_AR_THRESH)
                COUNTER = 0 # Prevent spam triggered alarms and force reset
        else:
            COUNTER = 0
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Yawning Check ──
        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "Yawning", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            cv2.putText(frame, f"Yawn Count: {yawns + 1}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawnStatus = False

        if prev_yawn_status and not yawnStatus:
            yawns += 1

        if yawns > 2:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not calibration_mode:
                threading.Thread(name='sound_yawning_alarm', target=sound_yawning_alarm).start()
                logger.log_event("yawn_detected", drive_minutes, mouEAR, MOU_AR_THRESH)
            yawns = 0

        cv2.putText(frame, f"MAR: {mouEAR:.2f}", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup & Logging ──
cap.release()
cv2.destroyAllWindows()
print("[INFO] Stream ended.")

total_drive_time = round((time.time() - drive_start_time) / 60, 2)
logger.save_and_print_summary(total_drive_time)
