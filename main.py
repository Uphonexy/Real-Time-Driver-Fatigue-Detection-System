import cv2
import dlib
import imutils
import time
from pygame import mixer
from imutils import face_utils

# Import separated modules
from detector import get_head_pose, eye_aspect_ratio, mouth_aspect_ratio, line_pairs
from thresholds import compute_adaptive_threshold, compute_adaptive_mar_threshold
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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

# ──────────────────────────────────────────────
# Pre-Detection Input Screen
# ──────────────────────────────────────────────
age_group, drive_start_time = run_calibration_phase(cap)

# ──────────────────────────────────────────────
# Threshold & System State Initialization
# ──────────────────────────────────────────────
personal_baseline_ear = 0.25
personal_baseline_mar = 0.65

# FIX A: Raised hardcoded fallback from 0.21 → 0.24
# Your face's closed-eye EAR (~0.25) was above the old 0.21 threshold,
# so "Eyes Closed" was never being detected.
EYE_AR_THRESH = 0.24
MOU_AR_THRESH = personal_baseline_mar

HEAD_PITCH_THRESH = 10
DISTRACTION_YAW_THRESH = 20

last_threshold_update = -1800

# Time-based eye closure tracking
eye_closed_start_time = None
EYE_CLOSED_DURATION_THRESH = 1.5  # seconds — must be closed this long to trigger
eye_event_logged = False

# Sustained yawn detection
yawn_start_time = None
YAWN_DURATION_THRESH = 1.5  # seconds — must be open this long to count as yawn
yawn_counted_this_open = False
yawnStatus = False
yawns = 0

# Face-loss counter
no_face_counter = 0

# ──────────────────────────────────────────────
# Session Logging & Calibration Initialization
# ──────────────────────────────────────────────
logger = SessionLogger(age_group)

calibration_mode = True
ear_samples = []
mar_samples = []
open_eye_avg = 0.0  # initialized to prevent NameError
calibration_end_time = 0
calibration_duration = 10  # seconds (5 minutes) — change to 10 for quick testing

# ──────────────────────────────────────────────
# Main Detection Loop
# ──────────────────────────────────────────────
print("[INFO] Running... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame.")
            break

        frame = imutils.resize(frame, width=640)
        frame_height, frame_width = frame.shape[:2]

        drive_minutes = (time.time() - drive_start_time) / 60
        hours = int(drive_minutes // 60)
        minutes = int(drive_minutes % 60)
        cv2.putText(frame, f"Drive Time: {hours:02d}:{minutes:02d}",
                    (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_drive_seconds = time.time() - drive_start_time

        # ── Calibration Phase ──
        if calibration_mode:
            remaining_time = int(calibration_duration - current_drive_seconds)
            if remaining_time > 0:
                cv2.putText(frame, "CALIBRATING - Sit normally, keep eyes open",
                            (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Calibrating: {remaining_time // 60}:{remaining_time % 60:02d} remaining",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                calibration_mode = False
                calibration_end_time = time.time()

                # EAR calibration — use top 70% of samples (open-eye values)
                if ear_samples:
                    ear_samples.sort(reverse=True)
                    top_70 = ear_samples[:int(len(ear_samples) * 0.7)]
                    if top_70:
                        open_eye_avg = sum(top_70) / len(top_70)
                        # FIX: reduced offset from 0.08 → 0.05 so threshold
                        # sits closer to the real open-eye EAR.
                        # Raised floor from 0.20 → 0.22 as a safer minimum.
                        personal_baseline_ear = max(0.22, open_eye_avg - 0.05)
                        # Apply immediately so detection works right after calibration
                        EYE_AR_THRESH = personal_baseline_ear
                        logger.update_baseline(personal_baseline_ear)

                # MAR calibration — use bottom 70% of samples (resting mouth values)
                if mar_samples:
                    mar_samples.sort()
                    bottom_70 = mar_samples[:int(len(mar_samples) * 0.7)]
                    if bottom_70:
                        closed_mouth_avg = sum(bottom_70) / len(bottom_70)
                        personal_baseline_mar = closed_mouth_avg + 0.10
                        MOU_AR_THRESH = personal_baseline_mar
                        logger.update_baseline_mar(personal_baseline_mar)

                print(f"[INFO] Calibration Complete.")
                print(f"       Open Eye Avg:   {open_eye_avg:.3f}")
                print(f"       EAR Threshold:  {EYE_AR_THRESH:.3f}  (if EAR drops below this → eyes closed)")
                print(f"       MAR Threshold:  {MOU_AR_THRESH:.3f}")
                last_threshold_update = -1800

        # ── Show calibration complete message for 5 seconds ──
        if not calibration_mode and time.time() - calibration_end_time < 5:
            cv2.putText(frame, f"Calibration Complete! EAR Thresh: {EYE_AR_THRESH:.3f}",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # ── Adaptive Threshold Update every 30 mins ──
        if not calibration_mode and current_drive_seconds - last_threshold_update >= 1800:
            EYE_AR_THRESH = compute_adaptive_threshold(personal_baseline_ear, age_group, drive_minutes)
            MOU_AR_THRESH = compute_adaptive_mar_threshold(personal_baseline_mar, age_group, drive_minutes)
            last_threshold_update = current_drive_seconds

        # ── Always show thresholds on HUD ──
        cv2.putText(frame, f"EAR Thresh: {EYE_AR_THRESH:.3f}",
                    (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        # ── Face-loss warning ──
        if len(rects) == 0:
            no_face_counter += 1
            if no_face_counter > 90:
                cv2.putText(frame, "FACE NOT DETECTED - Monitoring Paused",
                            (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            no_face_counter = 0

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

            # Collect samples during calibration
            if calibration_mode:
                ear_samples.append(ear)
                mar_samples.append(mouEAR)

            reprojectdst, euler_angle = get_head_pose(shape, frame_width, frame_height)

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Draw head pose box
            for start, end in line_pairs:
                cv2.line(frame,
                         tuple(map(int, reprojectdst[start])),
                         tuple(map(int, reprojectdst[end])),
                         (0, 0, 255))

            xx = euler_angle[0, 0]
            yy = euler_angle[1, 0]
            zz = euler_angle[2, 0]
            cv2.putText(frame, f"X: {xx:7.2f}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(frame, f"Y: {yy:7.2f}", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(frame, f"Z: {zz:7.2f}", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # ── Head Pose Check ──
            if xx > HEAD_PITCH_THRESH:
                cv2.putText(frame, "HEAD DOWN", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not calibration_mode:
                    sound_head_down_alarm()
                    logger.log_event("head_down", drive_minutes, xx, HEAD_PITCH_THRESH)

            # ── Distraction Check ──
            if abs(yy) > DISTRACTION_YAW_THRESH:
                cv2.putText(frame, "DISTRACTED", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if not calibration_mode:
                    sound_distracted_alarm()
                    logger.log_event("distracted", drive_minutes, yy, DISTRACTION_YAW_THRESH)

            # ── Eye Closure Check ──
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Eyes Closed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not calibration_mode:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    elapsed = time.time() - eye_closed_start_time

                    # Show live countdown timer on screen
                    cv2.putText(frame, f"Eye closed: {elapsed:.1f}s / {EYE_CLOSED_DURATION_THRESH}s",
                                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                    if elapsed >= EYE_CLOSED_DURATION_THRESH:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        sound_eyes_closed_alarm()
                        if not eye_event_logged:
                            logger.log_event("drowsiness_alarm", drive_minutes, ear, EYE_AR_THRESH)
                            eye_event_logged = True
            else:
                # Eyes are open — reset all closure tracking
                eye_closed_start_time = None
                eye_event_logged = False
                cv2.putText(frame, "Eyes Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ── Live EAR readout ──
            cv2.putText(frame, f"EAR: {ear:.3f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(frame, f"T: {EYE_AR_THRESH:.3f}", (480, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # ── Yawning Check ──
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                if time.time() - yawn_start_time >= YAWN_DURATION_THRESH:
                    yawnStatus = True
                    if not yawn_counted_this_open:
                        yawn_counted_this_open = True
            else:
                if yawnStatus:
                    yawns += 1
                    yawnStatus = False
                yawn_start_time = None
                yawn_counted_this_open = False

            # Show yawn count on HUD
            if yawns > 0:
                cv2.putText(frame, f"Yawns: {yawns}/3", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)

            if yawns >= 3:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not calibration_mode:
                    sound_yawning_alarm()
                    logger.log_event("yawn_detected", drive_minutes, mouEAR, MOU_AR_THRESH)
                yawns = 0

            # ── Live MAR readout ──
            cv2.putText(frame, f"MAR: {mouEAR:.3f}", (480, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Driver Fatigue Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream ended.")
    total_drive_time = round((time.time() - drive_start_time) / 60, 2)
    logger.save_and_print_summary(total_drive_time)