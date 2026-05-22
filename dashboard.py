import cv2
import dlib
import imutils
import time
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pygame import mixer
from imutils import face_utils
from datetime import datetime

try:
    from PIL import Image, ImageTk
except ImportError:
    print("\n[ERROR] Pillow library is required for the dashboard GUI.")
    print("Please install it using: pip install Pillow")
    sys.exit(1)

# Import separated modules exactly as main.py did
from detector import get_head_pose, eye_aspect_ratio, mouth_aspect_ratio, line_pairs
from thresholds import compute_adaptive_threshold, compute_adaptive_mar_threshold
from alarms import (
    sound_eyes_closed_alarm,
    sound_yawning_alarm,
    sound_head_down_alarm,
    sound_distracted_alarm
)
from logger import SessionLogger

# ── New modules ────────────────────────────────────────────────────────────
from database import init_db, create_session, get_sessions_for_driver
from driver_manager import run_driver_setup
from clip_recorder import ClipRecorder
from analytics import compute_risk_score, get_risk_label

# ──────────────────────────────────────────────
# Shared State
# ──────────────────────────────────────────────
state_lock = threading.Lock()
shared_state = {
    "frame": None,
    "ear": 0.0,
    "mar": 0.0,
    "eye_status": "Open",
    "mouth_status": "Closed",
    "face_detected": True,
    "yawns": 0,
    "drive_seconds": 0.0,
    "calibrating": True,
    "calib_remaining": 30,
    "paused": False,
    "stopped": False,
    "ear_thresh": 0.0,
    "mar_thresh": 0.0,
    "recent_alerts": [],
    "status_str": "CALIBRATING",
    "alert_active": False,
    "calibration_complete_time": 0,
    # New fields for pause-aware drive time
    "active_drive_seconds": 0.0,
}

# ──────────────────────────────────────────────
# Background Camera & Detection Thread
# ──────────────────────────────────────────────
def camera_thread_func(driver_id, driver_name, age_group, drive_start_time, session_id):
    mixer.init()
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

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
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS,          30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                time.sleep(3.0)

                success_count = 0
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if frame.mean() > 1.0:
                            success_count += 1

                if success_count >= 5:
                    print(f"[INFO] Camera opened successfully — {label}")
                    return cap
                else:
                    cap.release()
            except Exception:
                continue
        print("\n[ERROR] Could not open camera with any backend or index.")
        sys.exit(1)

    cap = open_camera()

    def read_frame_safe(cap, retries=3):
        for _ in range(retries):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
            time.sleep(0.05)
        return False, None

    # Logic Init
    personal_baseline_ear = 0.25
    personal_baseline_mar = 0.65
    EYE_AR_THRESH = 0.24
    MOU_AR_THRESH = personal_baseline_mar
    HEAD_PITCH_THRESH = 10
    DISTRACTION_YAW_THRESH = 20
    last_threshold_update = -1800
    eye_closed_start_time = None
    EYE_CLOSED_DURATION_THRESH = 1.5
    eye_event_logged = False
    yawn_start_time = None
    YAWN_DURATION_THRESH = 1.5
    yawn_counted_this_open = False
    yawnStatus = False
    yawns = 0
    no_face_counter = 0
    last_head_down_log  = 0.0
    last_distracted_log = 0.0
    LOG_COOLDOWN = 3.0

    # Pause tracking — mirrors main.py exactly (Bug fix)
    total_paused_seconds = 0.0
    pause_start_time_local = None

    logger = SessionLogger(age_group)
    if session_id is not None:
        logger.set_session_id(session_id)

    calibration_mode = True
    ear_samples = []
    mar_samples = []
    open_eye_avg = 0.0
    calibration_end_time = 0
    calibration_duration = 300  # BUG FIX: 5 minutes, matching main.py
    alert_clear_time = 0

    # Clip recorder
    clip_recorder = ClipRecorder()

    def add_alert(alert_type, description):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with state_lock:
            shared_state["recent_alerts"].insert(0, {"time": timestamp, "type": alert_type, "desc": description})
            if len(shared_state["recent_alerts"]) > 5:
                shared_state["recent_alerts"].pop()
            shared_state["alert_active"] = True
        nonlocal alert_clear_time
        alert_clear_time = time.time() + 3.0

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30

    try:
        while True:
            with state_lock:
                is_stopped = shared_state["stopped"]
                is_paused  = shared_state["paused"]

            if is_stopped:
                break

            ret, frame = read_frame_safe(cap)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                continue
            consecutive_failures = 0

            frame = imutils.resize(frame, width=640)
            frame_height, frame_width = frame.shape[:2]

            current_time = time.time()

            # ── BUG FIX: pause-aware active drive time ────────────────────
            if is_paused:
                # Track when this pause started (local to thread)
                if pause_start_time_local is None:
                    pause_start_time_local = current_time
            else:
                if pause_start_time_local is not None:
                    total_paused_seconds += current_time - pause_start_time_local
                    pause_start_time_local = None

            active_time = current_time - drive_start_time - total_paused_seconds
            if is_paused and pause_start_time_local is not None:
                active_time -= (current_time - pause_start_time_local)

            current_drive_seconds = max(0.0, active_time)
            drive_minutes = current_drive_seconds / 60

            # UI Update Prep
            ui_ear = 0.0
            ui_mar = 0.0
            ui_eye_status = "Open"
            ui_mouth_status = "Closed"
            ui_face_detected = True
            ui_status_str = "ACTIVE"
            ui_calib_remaining = 0

            # Handle Alerts Expiry
            if current_time > alert_clear_time:
                with state_lock:
                    shared_state["alert_active"] = False

            # Calibration Logic
            if calibration_mode and not is_paused:
                remaining_time = int(calibration_duration - current_drive_seconds)
                ui_calib_remaining = remaining_time
                if remaining_time > 0:
                    ui_status_str = "CALIBRATING"
                else:
                    calibration_mode = False
                    calibration_end_time = current_time
                    with state_lock:
                        shared_state["calibrating"] = False
                        shared_state["calibration_complete_time"] = current_time

                    if ear_samples:
                        ear_samples.sort(reverse=True)
                        top_70 = ear_samples[:int(len(ear_samples) * 0.7)]
                        if top_70:
                            open_eye_avg = sum(top_70) / len(top_70)
                            personal_baseline_ear = max(0.22, open_eye_avg - 0.05)
                            EYE_AR_THRESH = personal_baseline_ear
                            logger.update_baseline(personal_baseline_ear)

                    if mar_samples:
                        mar_samples.sort()
                        bottom_70 = mar_samples[:int(len(mar_samples) * 0.7)]
                        if bottom_70:
                            closed_mouth_avg = sum(bottom_70) / len(bottom_70)
                            personal_baseline_mar = closed_mouth_avg + 0.10
                            MOU_AR_THRESH = personal_baseline_mar
                            logger.update_baseline_mar(personal_baseline_mar)

                    last_threshold_update = -1800

            # Adaptive Threshold
            if not calibration_mode and current_drive_seconds - last_threshold_update >= 1800 and not is_paused:
                EYE_AR_THRESH = compute_adaptive_threshold(personal_baseline_ear, age_group, drive_minutes)
                MOU_AR_THRESH = compute_adaptive_mar_threshold(personal_baseline_mar, age_group, drive_minutes)
                last_threshold_update = current_drive_seconds

            # ── Clip recorder — feed frame every loop iteration ───────────
            if not calibration_mode and not is_paused:
                completed_clip = clip_recorder.add_frame(frame)
                if completed_clip:
                    print(f"[INFO] Clip saved: {completed_clip}")
            else:
                try:
                    import cv2 as _cv2
                    small = _cv2.resize(frame, (480, 360))
                    clip_recorder.buffer.append(small)
                except Exception:
                    pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            if len(rects) == 0:
                no_face_counter += 1
                ui_face_detected = False
                if no_face_counter > 90:
                    eye_closed_start_time = None
                    eye_event_logged = False
                    yawn_start_time = None
                    yawn_counted_this_open = False
                    yawnStatus = False
            else:
                no_face_counter = 0

            # Paused State Handling
            if is_paused:
                ui_status_str = "PAUSED"
                eye_closed_start_time = None
                yawn_start_time = None

            # Per-face Detection
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

                ui_ear = ear
                ui_mar = mouEAR

                if calibration_mode and not is_paused:
                    ear_samples.append(ear)
                    mar_samples.append(mouEAR)

                reprojectdst, euler_angle = get_head_pose(shape, frame_width, frame_height)

                # Draw landmarks on frame
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

                if not is_paused and not calibration_mode:
                    # Head Down
                    if xx > HEAD_PITCH_THRESH:
                        sound_head_down_alarm()
                        if current_time - last_head_down_log >= LOG_COOLDOWN:
                            logger.log_event("head_down", drive_minutes, xx, HEAD_PITCH_THRESH)
                            add_alert("head_down", "Head down")
                            last_head_down_log = current_time

                    # Distracted
                    if abs(yy) > DISTRACTION_YAW_THRESH:
                        sound_distracted_alarm()
                        if current_time - last_distracted_log >= LOG_COOLDOWN:
                            logger.log_event("distracted", drive_minutes, yy, DISTRACTION_YAW_THRESH)
                            add_alert("distracted", "Distracted")
                            last_distracted_log = current_time

                # Eyes closure
                if ear < EYE_AR_THRESH:
                    ui_eye_status = "Closed"
                    if not is_paused and not calibration_mode:
                        if eye_closed_start_time is None:
                            eye_closed_start_time = current_time
                        elapsed = current_time - eye_closed_start_time
                        if elapsed >= EYE_CLOSED_DURATION_THRESH:
                            ui_eye_status = "ALERT"
                            sound_eyes_closed_alarm()

                            # ── Clip recording trigger ──────────────────
                            clip_path = None
                            if not clip_recorder.is_recording():
                                try:
                                    clip_path = clip_recorder.start_recording()
                                except Exception as ce:
                                    print(f"[WARN] Clip recording failed: {ce}")
                                    clip_path = None

                            if not eye_event_logged:
                                logger.log_event(
                                    "drowsiness_alarm", drive_minutes, ear,
                                    EYE_AR_THRESH, clip_path=clip_path
                                )
                                add_alert("drowsiness_alarm", "Drowsiness detected")
                                eye_event_logged = True
                else:
                    eye_closed_start_time = None
                    eye_event_logged = False
                    ui_eye_status = "Open"

                # Yawning
                if mouEAR > MOU_AR_THRESH:
                    ui_mouth_status = "Yawning..."
                    if not is_paused:
                        if yawn_start_time is None:
                            yawn_start_time = current_time
                        yawn_elapsed = current_time - yawn_start_time
                        if yawn_elapsed >= YAWN_DURATION_THRESH:
                            yawnStatus = True
                            yawn_counted_this_open = True
                else:
                    ui_mouth_status = "Closed"
                    if yawnStatus:
                        yawns += 1
                        yawnStatus = False
                    yawn_start_time = None
                    yawn_counted_this_open = False

                if yawns >= 3 and not is_paused and not calibration_mode:
                    sound_yawning_alarm()
                    logger.log_event("yawn_detected", drive_minutes, mouEAR, MOU_AR_THRESH)
                    add_alert("yawn_detected", "3 yawns detected")
                    yawns = 0

            if not ui_face_detected:
                ui_eye_status = "N/A"
                ui_mouth_status = "N/A"

            if not calibration_mode and not is_paused:
                with state_lock:
                    if shared_state["alert_active"]:
                        ui_status_str = "ALERT"
                    else:
                        ui_status_str = "ACTIVE"
            elif calibration_mode and not is_paused:
                ui_status_str = "CALIBRATING"

            if calibration_mode and is_paused:
                ui_status_str = "PAUSED (Calib)"

            with state_lock:
                shared_state["frame"]                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shared_state["ear"]                     = ui_ear
                shared_state["mar"]                     = ui_mar
                shared_state["eye_status"]              = ui_eye_status
                shared_state["mouth_status"]            = ui_mouth_status
                shared_state["face_detected"]           = ui_face_detected
                shared_state["yawns"]                   = yawns
                shared_state["drive_seconds"]           = current_drive_seconds
                shared_state["active_drive_seconds"]    = current_drive_seconds
                shared_state["calib_remaining"]         = ui_calib_remaining
                shared_state["ear_thresh"]              = EYE_AR_THRESH
                shared_state["mar_thresh"]              = MOU_AR_THRESH
                shared_state["status_str"]              = ui_status_str

    finally:
        if cap is not None and cap.isOpened():
            cap.release()

        # Stop any in-progress clip recording
        try:
            if clip_recorder.is_recording():
                clip_recorder.stop_recording()
        except Exception:
            pass

        # Finalise pause seconds
        if pause_start_time_local is not None:
            total_paused_seconds += time.time() - pause_start_time_local

        total_drive_time = round((time.time() - drive_start_time - total_paused_seconds) / 60, 2)

        # Compute risk score
        risk_score = None
        if session_id is not None:
            try:
                risk_score = compute_risk_score(session_id)
            except Exception as e:
                print(f"[WARN] Could not compute risk score: {e}")

        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            logger.save_and_print_summary(
                total_drive_time,
                session_id=session_id,
                risk_score=risk_score,
            )
        out = f.getvalue()
        sys.stdout.write(out)

        fname = "session.json"
        for line in out.split('\n'):
            if "[INFO] Log saved to" in line:
                fname = line.split("Log saved to ")[1].strip()

        add_alert("system", f"Session saved: {fname}")
        print("[INFO] Stream ended.")


# ──────────────────────────────────────────────
# Dashboard UI Application
# ──────────────────────────────────────────────
class DashboardApp:
    def __init__(self, root, drive_start_time, driver_id, driver_name, age_group, session_id):
        self.root = root
        self.root.title("WAKEMATE — Driver Fatigue Detection")
        self.root.geometry("1150x680")
        self.root.configure(bg="#0a0f1e")
        self.root.resizable(False, False)

        self.drive_start_time = drive_start_time
        self.driver_id   = driver_id
        self.driver_name = driver_name
        self.age_group   = age_group
        self.session_id  = session_id

        # Colors
        self.C_BG    = "#0a0f1e"
        self.C_PANEL = "#111827"
        self.C_CYAN  = "#00d4ff"
        self.C_RED   = "#ff3b3b"
        self.C_GREEN = "#00e676"
        self.C_AMBER = "#ffaa00"
        self.C_WHITE = "#ffffff"
        self.C_GRAY  = "#888888"

        # Toggles for flashing
        self.flash_toggle = False
        self.calib_pulse = 0
        self.calib_dir = 1

        self.setup_ui()

        # Start Update Loops
        self.root.after(30, self.update_feed)
        self.root.after(100, self.update_stats)
        self.root.after(500, self.toggle_flash)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_flash(self):
        self.flash_toggle = not self.flash_toggle
        self.root.after(500, self.toggle_flash)

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.C_BG)
        header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(header_frame, text="WAKEMATE  🚗  Driver Fatigue Detection",
                 font=("Segoe UI", 18, "bold"), bg=self.C_BG, fg=self.C_WHITE).pack(side=tk.LEFT)

        tk.Label(header_frame,
                 text=f"Driver: {self.driver_name}  |  Age: {self.age_group}",
                 font=("Segoe UI", 11), bg=self.C_BG, fg=self.C_GRAY).pack(side=tk.RIGHT)

        # Tab bar
        self.tab_bar = tk.Frame(self.root, bg=self.C_BG)
        self.tab_bar.pack(fill=tk.X, padx=20, pady=(8, 0))

        self.tab_content = tk.Frame(self.root, bg=self.C_BG)
        self.tab_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        self._tab_panels: dict[str, tk.Frame] = {}
        self._tab_btns:   dict[str, tk.Button] = {}

        self._build_live_panel()
        self._build_history_panel()

        self._show_tab("live")

        # ── CONTROL PANEL ──
        ctrl_frame = tk.Frame(self.root, bg=self.C_PANEL)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=8)

        tk.Label(ctrl_frame, text="CONTROL PANEL:", font=("Segoe UI", 12, "bold"),
                 bg=self.C_PANEL, fg=self.C_WHITE).pack(side=tk.LEFT, padx=10)

        self.btn_start = tk.Button(ctrl_frame, text="▶ Start", font=("Segoe UI", 11, "bold"),
                                   bg=self.C_PANEL, fg=self.C_GREEN, bd=2, relief=tk.RAISED,
                                   command=self.cmd_start)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = tk.Button(ctrl_frame, text="⏸ Pause", font=("Segoe UI", 11, "bold"),
                                   bg=self.C_PANEL, fg=self.C_AMBER, bd=2, relief=tk.RAISED,
                                   command=self.cmd_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(ctrl_frame, text="⏹ Stop", font=("Segoe UI", 11, "bold"),
                                  bg=self.C_PANEL, fg=self.C_RED, bd=2, relief=tk.RAISED,
                                  command=self.cmd_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_frame, text="🔄 Refresh History", font=("Segoe UI", 11),
                  bg=self.C_PANEL, fg=self.C_CYAN, bd=1, relief=tk.FLAT,
                  command=self.refresh_history).pack(side=tk.RIGHT, padx=10)

        tk.Label(self.root, text="Press buttons to control detection (video keeps streaming).",
                 font=("Segoe UI", 10), bg=self.C_BG, fg=self.C_GRAY).pack(pady=0)

    # ──────────────────────────────────────────
    # Tab management
    # ──────────────────────────────────────────
    def _make_tab_btn(self, label: str, tab_name: str):
        btn = tk.Button(
            self.tab_bar, text=label,
            font=("Segoe UI", 11, "bold"),
            bg=self.C_PANEL, fg=self.C_GRAY,
            bd=0, relief=tk.FLAT, padx=14, pady=5,
            activebackground=self.C_PANEL, activeforeground=self.C_CYAN,
            command=lambda: self._show_tab(tab_name),
        )
        btn.pack(side=tk.LEFT, padx=2)
        self._tab_btns[tab_name] = btn

    def _show_tab(self, name: str):
        for n, panel in self._tab_panels.items():
            panel.pack_forget()
        self._tab_panels[name].pack(fill=tk.BOTH, expand=True)
        for n, btn in self._tab_btns.items():
            btn.config(fg=self.C_CYAN if n == name else self.C_GRAY)
        if name == "history":
            self.refresh_history()

    # ──────────────────────────────────────────
    # Live monitoring panel (original 3-column layout)
    # ──────────────────────────────────────────
    def _build_live_panel(self):
        self._make_tab_btn("📷 Live Monitor", "live")

        panel = tk.Frame(self.tab_content, bg=self.C_BG)
        self._tab_panels["live"] = panel

        main_frame = tk.Frame(panel, bg=self.C_BG)
        main_frame.pack(fill=tk.BOTH, expand=True)

        col1 = tk.Frame(main_frame, bg=self.C_PANEL, width=500, height=450)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        col1.pack_propagate(False)

        col2 = tk.Frame(main_frame, bg=self.C_PANEL, width=290, height=450)
        col2.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        col2.pack_propagate(False)

        col3 = tk.Frame(main_frame, bg=self.C_PANEL, width=290, height=450)
        col3.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        col3.pack_propagate(False)

        # ── COLUMN 1: LIVE FEED ──
        tk.Label(col1, text="📷 Live Feed", font=("Segoe UI", 14, "bold"),
                 bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        self.feed_border = tk.Frame(col1, bg=self.C_PANEL, bd=3)
        self.feed_border.pack(pady=5)
        self.video_label = tk.Label(self.feed_border, bg="#000")
        self.video_label.pack()

        # ── COLUMN 2: DRIVER STATS ──
        tk.Label(col2, text="👤 Driver Stats", font=("Segoe UI", 14, "bold"),
                 bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        self.stats = {}
        rows = [
            ("Face detected", "✓ Active"),
            ("Eyes", "Open"),
            ("Mouth", "Closed"),
            ("EAR", "0.000"),
            ("MAR", "0.000"),
            ("EAR Threshold", "0.000"),
            ("Yawns", "0 / 3"),
            ("Drive time", "00:00:00"),
            ("Status", "ACTIVE"),
        ]
        stats_frame = tk.Frame(col2, bg=self.C_PANEL)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        for name, default in rows:
            row_frame = tk.Frame(stats_frame, bg=self.C_PANEL)
            row_frame.pack(fill=tk.X, pady=5)
            tk.Label(row_frame, text=name, font=("Segoe UI", 12), bg=self.C_PANEL, fg=self.C_WHITE).pack(side=tk.LEFT)
            lbl_val = tk.Label(row_frame, text=default, font=("Consolas", 12, "bold"), bg=self.C_PANEL, fg=self.C_CYAN)
            lbl_val.pack(side=tk.RIGHT)
            self.stats[name] = lbl_val

        self.lbl_calib_banner = tk.Label(col2, text="", font=("Segoe UI", 12, "bold"), bg=self.C_PANEL, fg=self.C_GREEN)
        self.lbl_calib_banner.pack(pady=5)

        # ── COLUMN 3: ALERTS ──
        tk.Label(col3, text="⚠ Alerts (Last 5)", font=("Segoe UI", 14, "bold"),
                 bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        self.alerts_frame = tk.Frame(col3, bg=self.C_PANEL)
        self.alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.alert_widgets = []
        for _ in range(5):
            frm = tk.Frame(self.alerts_frame, bg=self.C_PANEL)
            frm.pack(fill=tk.X, pady=3)
            border = tk.Frame(frm, bg=self.C_PANEL, width=4)
            border.pack(side=tk.LEFT, fill=tk.Y)
            lbl = tk.Label(frm, text="", font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_GRAY, anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, padx=5)
            self.alert_widgets.append((border, lbl))

        start_time_str = datetime.fromtimestamp(self.drive_start_time).strftime('%H:%M:%S')
        tk.Label(col3, text=f"Session started: {start_time_str}", font=("Segoe UI", 10),
                 bg=self.C_PANEL, fg=self.C_GRAY).pack(side=tk.BOTTOM, pady=10)

    # ──────────────────────────────────────────
    # History panel (4th tab — new)
    # ──────────────────────────────────────────
    def _build_history_panel(self):
        self._make_tab_btn("📊 History", "history")

        panel = tk.Frame(self.tab_content, bg=self.C_PANEL)
        self._tab_panels["history"] = panel

        tk.Label(panel, text="📊 Session History",
                 font=("Segoe UI", 15, "bold"), bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=(12, 4))
        tk.Label(panel, text=f"Driver: {self.driver_name}  |  All past sessions",
                 font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_GRAY).pack(pady=(0, 8))

        # Treeview with style
        style = ttk.Style()
        style.theme_use("default")
        style.configure("History.Treeview",
                        background=self.C_PANEL,
                        foreground=self.C_WHITE,
                        fieldbackground=self.C_PANEL,
                        rowheight=26,
                        font=("Segoe UI", 10))
        style.configure("History.Treeview.Heading",
                        background="#1e293b",
                        foreground=self.C_CYAN,
                        font=("Segoe UI", 10, "bold"),
                        relief="flat")
        style.map("History.Treeview", background=[("selected", "#1e3a5f")])

        cols = ("Date", "Drive Time", "Risk Score", "Drowsy", "Yawns", "Head Down", "Distracted")

        tree_frame = tk.Frame(panel, bg=self.C_PANEL)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            style="History.Treeview",
            yscrollcommand=vsb.set,
        )
        vsb.config(command=self.history_tree.yview)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        col_widths = [120, 110, 130, 80, 80, 110, 110]
        for col, w in zip(cols, col_widths):
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=w, anchor="center")

        # Tag colours for risk levels
        self.history_tree.tag_configure("low",      foreground="#00e676")
        self.history_tree.tag_configure("moderate", foreground="#ffaa00")
        self.history_tree.tag_configure("high",     foreground="#ff3b3b")
        self.history_tree.tag_configure("critical", foreground="#ff3b3b", font=("Segoe UI", 10, "bold"))

    def refresh_history(self):
        """Reload session history from DB into the Treeview."""
        try:
            for row in self.history_tree.get_children():
                self.history_tree.delete(row)

            sessions = get_sessions_for_driver(self.driver_id)
            if not sessions:
                self.history_tree.insert("", "end", values=("No sessions yet", "", "", "", "", "", ""))
                return

            for s in sessions:
                sid        = s.get("id", 0)
                start      = (s.get("start_time") or "")[:16].replace("T", "  ")
                drive_mins = s.get("total_drive_mins") or 0.0
                drive_str  = f"{round(drive_mins, 1)} min"

                # Count events
                try:
                    from database import get_events_for_session
                    evts = get_events_for_session(sid)
                    def cnt(t): return sum(1 for e in evts if e.get("event_type") == t)
                    n_drowsy  = cnt("drowsiness_alarm")
                    n_yawn    = cnt("yawn_detected")
                    n_head    = cnt("head_down")
                    n_distr   = cnt("distracted")
                except Exception:
                    n_drowsy = n_yawn = n_head = n_distr = "?"

                # Risk score
                try:
                    score = compute_risk_score(sid)
                    label = get_risk_label(score)
                    risk_str = f"{score}  {label}"
                except Exception:
                    score = 0.0
                    label = "LOW RISK"
                    risk_str = "N/A"

                # Tag for colouring
                tag = "low"
                if label == "MODERATE RISK":
                    tag = "moderate"
                elif label == "HIGH RISK":
                    tag = "high"
                elif label == "CRITICAL RISK":
                    tag = "critical"

                self.history_tree.insert(
                    "", "end",
                    values=(start, drive_str, risk_str, n_drowsy, n_yawn, n_head, n_distr),
                    tags=(tag,),
                )
        except Exception as e:
            print(f"[Dashboard WARNING] refresh_history failed: {e}")

    # ──────────────────────────────────────────
    # Control buttons
    # ──────────────────────────────────────────
    def cmd_start(self):
        with state_lock:
            shared_state["paused"] = False

    def cmd_pause(self):
        with state_lock:
            shared_state["paused"] = True

    def cmd_stop(self):
        with state_lock:
            shared_state["stopped"] = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)

    def on_closing(self):
        self.cmd_stop()
        self.root.after(500, self.root.destroy)

    # ──────────────────────────────────────────
    # Update loops
    # ──────────────────────────────────────────
    def update_feed(self):
        with state_lock:
            frame_rgb     = shared_state["frame"]
            is_calibrating = shared_state["calibrating"]
            calib_rem     = shared_state["calib_remaining"]
            alert_active  = shared_state["alert_active"]

        if frame_rgb is not None:
            img = Image.fromarray(frame_rgb)
            img = img.resize((480, 360), Image.Resampling.LANCZOS)

            if is_calibrating:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                txt = f"CALIBRATING - {calib_rem}s left"
                draw.text((10, 330), txt, fill=(0, 212, 255))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if alert_active:
            self.feed_border.config(bg=self.C_RED if self.flash_toggle else self.C_PANEL)
        elif is_calibrating:
            self.feed_border.config(bg=self.C_CYAN if self.flash_toggle else self.C_PANEL)
        else:
            self.feed_border.config(bg=self.C_PANEL)

        self.root.after(30, self.update_feed)

    def update_stats(self):
        with state_lock:
            st = shared_state.copy()

        # ── BUG FIX: use pause-aware active_drive_seconds ──
        active_secs = st.get("active_drive_seconds", st["drive_seconds"])
        hrs  = int(active_secs // 3600)
        mins = int((active_secs % 3600) // 60)
        secs = int(active_secs % 60)
        dr_time = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        # Face
        if st["face_detected"]:
            self.stats["Face detected"].config(text="✓ Active", fg=self.C_GREEN)
        else:
            self.stats["Face detected"].config(text="✗ Not detected",
                                               fg=self.C_RED if self.flash_toggle else self.C_PANEL)

        # Eyes
        eye_color = self.C_GREEN
        if st["eye_status"] == "Closed":
            eye_color = self.C_RED
        elif st["eye_status"] == "ALERT":
            eye_color = self.C_RED if self.flash_toggle else self.C_PANEL
        elif st["eye_status"] == "N/A":
            eye_color = self.C_GRAY
        self.stats["Eyes"].config(text=st["eye_status"], fg=eye_color)

        # Mouth
        m_color = self.C_GREEN
        if st["mouth_status"] == "Yawning...":
            m_color = self.C_AMBER if self.flash_toggle else self.C_PANEL
        elif st["mouth_status"] == "N/A":
            m_color = self.C_GRAY
        self.stats["Mouth"].config(text=st["mouth_status"], fg=m_color)

        # EAR
        ear_color = self.C_CYAN
        if st["ear"] < st["ear_thresh"]:
            ear_color = self.C_RED
        self.stats["EAR"].config(text=f"{st['ear']:.3f}", fg=ear_color)

        # MAR
        mar_color = self.C_CYAN
        if st["mar"] > st["mar_thresh"]:
            mar_color = self.C_AMBER
        self.stats["MAR"].config(text=f"{st['mar']:.3f}", fg=mar_color)

        # Thresholds
        self.stats["EAR Threshold"].config(text=f"{st['ear_thresh']:.3f}", fg=self.C_WHITE)

        # Yawns
        y_color = self.C_WHITE
        if st["yawns"] > 0:
            y_color = self.C_AMBER
        if st["yawns"] >= 3:
            y_color = self.C_RED
        self.stats["Yawns"].config(text=f"{st['yawns']} / 3", fg=y_color)

        # Drive time (now pause-corrected)
        self.stats["Drive time"].config(text=dr_time, fg=self.C_WHITE)

        # Status
        s_color = self.C_GREEN
        if "PAUSED" in st["status_str"]:
            s_color = self.C_AMBER
        elif "CALIBRATING" in st["status_str"]:
            s_color = self.C_CYAN if self.flash_toggle else self.C_PANEL
        elif "ALERT" in st["status_str"]:
            s_color = self.C_RED if self.flash_toggle else self.C_PANEL
        self.stats["Status"].config(text=st["status_str"], fg=s_color)

        # Calib Banner
        if not st["calibrating"] and time.time() - st["calibration_complete_time"] < 3:
            self.lbl_calib_banner.config(text="✓ Calibration complete")
        else:
            self.lbl_calib_banner.config(text="")

        # Alerts Panel
        if not st["recent_alerts"]:
            self.alert_widgets[0][1].config(text="None", fg=self.C_GRAY)
            self.alert_widgets[0][0].config(bg=self.C_PANEL)
            for bw, lw in self.alert_widgets[1:]:
                lw.config(text="")
                bw.config(bg=self.C_PANEL)
        else:
            colors = {
                "drowsiness_alarm": self.C_RED,
                "yawn_detected":    self.C_AMBER,
                "head_down":        self.C_AMBER,
                "distracted":       self.C_CYAN,
                "system":           self.C_GREEN,
            }
            for i in range(5):
                if i < len(st["recent_alerts"]):
                    al = st["recent_alerts"][i]
                    c  = colors.get(al["type"], self.C_WHITE)
                    self.alert_widgets[i][1].config(text=f"{al['time']}  {al['desc']}", fg=self.C_WHITE)
                    self.alert_widgets[i][0].config(bg=c)
                else:
                    self.alert_widgets[i][1].config(text="")
                    self.alert_widgets[i][0].config(bg=self.C_PANEL)

        self.root.after(100, self.update_stats)


def main():
    # Initialise DB first
    init_db()

    # Driver setup dialog
    driver_id, driver_name, age_group = run_driver_setup()
    if driver_id is None:
        print("[INFO] Driver setup aborted.")
        sys.exit(0)

    drive_start_time = time.time()

    # Create a new session row
    session_id = create_session(driver_id, age_group, datetime.now().isoformat())
    print(f"[INFO] Driver: {driver_name}  |  Age: {age_group}  |  Session ID: {session_id}")

    bg_thread = threading.Thread(
        target=camera_thread_func,
        args=(driver_id, driver_name, age_group, drive_start_time, session_id),
        daemon=True,
    )
    bg_thread.start()

    root = tk.Tk()
    app = DashboardApp(root, drive_start_time, driver_id, driver_name, age_group, session_id)
    root.mainloop()


if __name__ == "__main__":
    main()
