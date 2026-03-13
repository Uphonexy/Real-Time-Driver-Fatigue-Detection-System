import cv2
import dlib
import imutils
import time
import sys
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

# ──────────────────────────────────────────────
# Bulletproof Camera Opener
# Tries every backend + index combination until
# one actually delivers real, non-black frames.
# Sleep increased to 3.0s and success threshold
# raised to 5 frames so slow laptop cameras have
# enough time to fully initialize before we check.
# BUFFERSIZE=1 flushes stale/black frames instantly.
# ──────────────────────────────────────────────
def open_camera():
    print("[INFO] Starting video stream...")

    attempts = [
        (0, cv2.CAP_DSHOW, "index 0 + DirectShow"),
        (1, cv2.CAP_DSHOW, "index 1 + DirectShow"),
        (2, cv2.CAP_DSHOW, "index 2 + DirectShow"),
        (0, cv2.CAP_MSMF,  "index 0 + MSMF"),
        (1, cv2.CAP_MSMF,  "index 1 + MSMF"),
        (0, cv2.CAP_ANY,   "index 0 + Auto"),
        (1, cv2.CAP_ANY,   "index 1 + Auto"),
    ]

    for idx, backend, label in attempts:
        print(f"[INFO] Trying camera {label}...")
        try:
            cap = cv2.VideoCapture(idx, backend)

            if not cap.isOpened():
                cap.release()
                print(f"[WARN] {label} — could not open.")
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # flush stale/black frames

            # Give camera hardware 3s to fully wake up
            # (was 1.5s — too short for many laptop cameras)
            time.sleep(3.0)

            # Drain warm-up frames — first several are often black on Windows
            success_count = 0
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Extra check: reject pure-black frames
                    if frame.mean() > 1.0:
                        success_count += 1

            # Raised from 3 → 5 to ensure camera is truly stable
            if success_count >= 5:
                print(f"[INFO] Camera opened successfully — {label}")
                return cap
            else:
                cap.release()
                print(f"[WARN] {label} — opened but frames black/unreadable ({success_count}/10).")

        except Exception as e:
            print(f"[WARN] {label} — exception: {e}")
            continue

    print("\n[ERROR] Could not open camera with any backend or index.")
    print("        Please check:")
    print("        1. Close ALL apps using your webcam (Teams, Zoom, Discord, Camera app)")
    print("        2. Windows Settings → Privacy → Camera → ensure access is ON")
    print("        3. Unplug and replug webcam if external")
    print("        4. Restart your PC and try again")
    sys.exit(1)


cap = open_camera()

# ──────────────────────────────────────────────
# Pre-Detection Input Screen (Age Group Selection)
# ──────────────────────────────────────────────
age_group, drive_start_time = run_calibration_phase(cap)

# ──────────────────────────────────────────────
# Threshold & System State Initialization
# ──────────────────────────────────────────────
personal_baseline_ear = 0.25
personal_baseline_mar = 0.65

# Fallback raised from 0.21 → 0.24 because most faces
# have a closed-eye EAR around 0.20–0.25 and 0.21 was
# too low to catch actual eye closures.
EYE_AR_THRESH = 0.24
MOU_AR_THRESH = personal_baseline_mar

HEAD_PITCH_THRESH      = 10   # degrees — head nodding forward
DISTRACTION_YAW_THRESH = 20   # degrees — head turning sideways

last_threshold_update = -1800  # force immediate update after calibration

# ── Eye closure tracking (time-based, not frame-count-based) ──
eye_closed_start_time      = None
EYE_CLOSED_DURATION_THRESH = 1.5   # seconds closed before alert fires
eye_event_logged           = False

# ── Yawn tracking (requires sustained mouth open) ──
yawn_start_time        = None
YAWN_DURATION_THRESH   = 1.5   # seconds open before it counts as a real yawn
yawn_counted_this_open = False
yawnStatus             = False
yawns                  = 0

# ── Face-loss tracking ──
no_face_counter = 0

# ──────────────────────────────────────────────
# Session Logger & Calibration Setup
# ──────────────────────────────────────────────
logger = SessionLogger(age_group)

calibration_mode     = True
ear_samples          = []
mar_samples          = []
open_eye_avg         = 0.0   # safe default — prevents NameError if calibration skipped
calibration_end_time = 0
calibration_duration = 30    # seconds — change to 300 for full 5-min production use

# ──────────────────────────────────────────────
# Safe Frame Reader with Auto-Retry
# Single dropped frames won't crash the session
# ──────────────────────────────────────────────
def read_frame_safe(cap, retries=3):
    for _ in range(retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.05)
    return False, None


# ──────────────────────────────────────────────
# Main Detection Loop
# ──────────────────────────────────────────────
print("[INFO] Running... Press 'q' to quit.")

consecutive_failures     = 0
MAX_CONSECUTIVE_FAILURES = 30  # ~1 second of failures triggers exit

try:
    while True:
        ret, frame = read_frame_safe(cap)

        if not ret:
            consecutive_failures += 1
            print(f"[WARN] Frame read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("[ERROR] Too many consecutive frame failures. Exiting.")
                break
            continue

        consecutive_failures = 0  # reset on success

        frame = imutils.resize(frame, width=640)
        frame_height, frame_width = frame.shape[:2]

        drive_minutes = (time.time() - drive_start_time) / 60
        hours   = int(drive_minutes // 60)
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
                cv2.putText(frame, f"Time left: {remaining_time // 60}:{remaining_time % 60:02d}",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                calibration_mode     = False
                calibration_end_time = time.time()

                # EAR calibration: top 70% = open-eye values
                if ear_samples:
                    ear_samples.sort(reverse=True)
                    top_70 = ear_samples[:int(len(ear_samples) * 0.7)]
                    if top_70:
                        open_eye_avg          = sum(top_70) / len(top_70)
                        # Offset reduced 0.08 → 0.05 so threshold sits close
                        # enough to your actual closed-eye EAR to catch it.
                        # Floor raised 0.20 → 0.22 as a safer minimum.
                        personal_baseline_ear = max(0.22, open_eye_avg - 0.05)
                        EYE_AR_THRESH         = personal_baseline_ear
                        logger.update_baseline(personal_baseline_ear)

                # MAR calibration: bottom 70% = resting-mouth values
                if mar_samples:
                    mar_samples.sort()
                    bottom_70 = mar_samples[:int(len(mar_samples) * 0.7)]
                    if bottom_70:
                        closed_mouth_avg      = sum(bottom_70) / len(bottom_70)
                        personal_baseline_mar = closed_mouth_avg + 0.10
                        MOU_AR_THRESH         = personal_baseline_mar
                        logger.update_baseline_mar(personal_baseline_mar)

                print(f"[INFO] Calibration Complete.")
                print(f"       Open Eye Avg  : {open_eye_avg:.3f}")
                print(f"       EAR Threshold : {EYE_AR_THRESH:.3f}  <- EAR must drop below this")
                print(f"       MAR Threshold : {MOU_AR_THRESH:.3f}")
                last_threshold_update = -1800

        # Show calibration complete banner for 5 seconds
        if not calibration_mode and time.time() - calibration_end_time < 5:
            cv2.putText(frame, f"Calibration Complete!  EAR Thresh: {EYE_AR_THRESH:.3f}",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # ── Adaptive Threshold Update every 30 minutes ──
        if not calibration_mode and current_drive_seconds - last_threshold_update >= 1800:
            EYE_AR_THRESH = compute_adaptive_threshold(
                personal_baseline_ear, age_group, drive_minutes)
            MOU_AR_THRESH = compute_adaptive_mar_threshold(
                personal_baseline_mar, age_group, drive_minutes)
            last_threshold_update = current_drive_seconds
            print(f"[INFO] Thresholds updated — EAR: {EYE_AR_THRESH:.3f}, MAR: {MOU_AR_THRESH:.3f}")

        # Always show current thresholds on HUD
        cv2.putText(frame, f"EAR Thresh: {EYE_AR_THRESH:.3f}",
                    (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        # ── Face-loss warning ──
        if len(rects) == 0:
            no_face_counter += 1
            if no_face_counter > 90:  # ~3 seconds at 30fps
                cv2.putText(frame, "FACE NOT DETECTED - Monitoring Paused",
                            (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                # Reset all timers so stale state doesn't carry over
                eye_closed_start_time  = None
                eye_event_logged       = False
                yawn_start_time        = None
                yawn_counted_this_open = False
                yawnStatus             = False
        else:
            no_face_counter = 0

        # ── Per-face Detection ──
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

            # Collect samples during calibration only
            if calibration_mode:
                ear_samples.append(ear)
                mar_samples.append(mouEAR)

            reprojectdst, euler_angle = get_head_pose(shape, frame_width, frame_height)

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Draw head pose axes
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

            # ────────────────────────────────────────
            # HEAD DOWN CHECK
            # ────────────────────────────────────────
            if xx > HEAD_PITCH_THRESH:
                cv2.putText(frame, "HEAD DOWN", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not calibration_mode:
                    sound_head_down_alarm()
                    logger.log_event("head_down", drive_minutes, xx, HEAD_PITCH_THRESH)

            # ────────────────────────────────────────
            # DISTRACTION CHECK
            # ────────────────────────────────────────
            if abs(yy) > DISTRACTION_YAW_THRESH:
                cv2.putText(frame, "DISTRACTED", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if not calibration_mode:
                    sound_distracted_alarm()
                    logger.log_event("distracted", drive_minutes, yy, DISTRACTION_YAW_THRESH)

            # ────────────────────────────────────────
            # EYE CLOSURE CHECK
            # Time-based — works correctly at any FPS
            # ────────────────────────────────────────
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Eyes Closed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not calibration_mode:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()

                    elapsed = time.time() - eye_closed_start_time

                    # Live closure timer — visible on screen so you can confirm it works
                    cv2.putText(frame,
                                f"Eye closed: {elapsed:.1f}s / {EYE_CLOSED_DURATION_THRESH:.1f}s",
                                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                    if elapsed >= EYE_CLOSED_DURATION_THRESH:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                        sound_eyes_closed_alarm()
                        if not eye_event_logged:
                            logger.log_event("drowsiness_alarm", drive_minutes, ear, EYE_AR_THRESH)
                            eye_event_logged = True
            else:
                # Eyes open — reset all closure tracking
                eye_closed_start_time = None
                eye_event_logged      = False
                cv2.putText(frame, "Eyes Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # EAR live readout
            cv2.putText(frame, f"EAR: {ear:.3f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(frame, f"T:   {EYE_AR_THRESH:.3f}", (480, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # ────────────────────────────────────────
            # YAWN CHECK
            # Requires mouth open >= 1.5s continuously
            # Talking / deep breaths won't trigger it
            # ────────────────────────────────────────
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if yawn_start_time is None:
                    yawn_start_time = time.time()

                yawn_elapsed = time.time() - yawn_start_time
                cv2.putText(frame, f"Yawn: {yawn_elapsed:.1f}s",
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                if yawn_elapsed >= YAWN_DURATION_THRESH:
                    yawnStatus             = True
                    yawn_counted_this_open = True
            else:
                if yawnStatus:
                    yawns     += 1
                    yawnStatus  = False
                yawn_start_time        = None
                yawn_counted_this_open = False

            # Yawn progress counter on HUD
            if yawns > 0:
                cv2.putText(frame, f"Yawns: {yawns}/3",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)

            if yawns >= 3:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                if not calibration_mode:
                    sound_yawning_alarm()
                    logger.log_event("yawn_detected", drive_minutes, mouEAR, MOU_AR_THRESH)
                yawns = 0

            # MAR live readout
            cv2.putText(frame, f"MAR: {mouEAR:.3f}", (480, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Driver Fatigue Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always runs — even on Ctrl+C or unexpected crash
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream ended.")
    total_drive_time = round((time.time() - drive_start_time) / 60, 2)
    logger.save_and_print_summary(total_drive_time)