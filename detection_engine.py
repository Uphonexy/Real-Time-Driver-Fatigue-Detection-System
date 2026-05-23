"""
detection_engine.py — Core fatigue detection logic for WakeMate v2.0.
Extracted into a single class to avoid duplication between main.py and dashboard.py.
"""
from __future__ import annotations

import os
import time
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

# ── Internal modules ─────────────────────────────────────────────────────────
from app_logger import get_logger
from detector import get_head_pose, eye_aspect_ratio, mouth_aspect_ratio, line_pairs
from thresholds import compute_adaptive_threshold, compute_adaptive_mar_threshold
from alarms import (
    sound_eyes_closed_alarm,
    sound_yawning_alarm,
    sound_head_down_alarm,
    sound_distracted_alarm,
)
from logger import SessionLogger
from clip_recorder import ClipRecorder, cleanup_old_clips

_log = get_logger("engine")

# ── Tunable constants ─────────────────────────────────────────────────────────
CALIBRATION_DURATION       = 30    # 30 seconds for quick setup
EYE_CLOSED_DURATION_THRESH = 1.5   # seconds before drowsiness alarm
YAWN_DURATION_THRESH       = 1.5   # seconds of continuous open mouth
HEAD_DOWN_DURATION_THRESH  = 1.0   # seconds of continuous head down before alarm
DISTRACTED_DURATION_THRESH = 1.0   # seconds of continuous distraction before alarm
NO_FACE_FRAME_LIMIT        = 90    # frames before "face not detected" banner
MAX_CONSECUTIVE_FAILURES   = 30    # consecutive bad reads before giving up
LOG_COOLDOWN               = 3.0   # seconds between repeated event DB writes
ADAPTIVE_THRESH_INTERVAL   = 1800  # seconds between adaptive recalculations
BREAK_REMINDER_MINUTES     = 120   # drive time before break reminder fires
YAWN_WINDOW_SECS           = 600   # 10-minute sliding window

# Camera attempts (indices 0-2 × CAP_DSHOW / CAP_MSMF / CAP_ANY)
_CAMERA_ATTEMPTS = [
    (0, cv2.CAP_DSHOW, "index 0 + DirectShow"),
    (1, cv2.CAP_DSHOW, "index 1 + DirectShow"),
    (2, cv2.CAP_DSHOW, "index 2 + DirectShow"),
    (0, cv2.CAP_MSMF,  "index 0 + MSMF"),
    (1, cv2.CAP_MSMF,  "index 1 + MSMF"),
    (0, cv2.CAP_ANY,   "index 0 + Auto"),
    (1, cv2.CAP_ANY,   "index 1 + Auto"),
]

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """All information the UI layer needs after one detection cycle."""
    frame:                np.ndarray
    ear:                  float = 0.0
    mar:                  float = 0.0
    ear_thresh:           float = 0.0
    mar_thresh:           float = 0.0
    eye_status:           str   = "Open"
    mouth_status:         str   = "Closed"
    status_str:           str   = "ACTIVE"
    face_detected:        bool  = True
    calibrating:          bool  = True
    calib_remaining:      int   = 0
    calibration_complete: bool  = False
    yawns:                int   = 0
    drive_seconds:        float = 0.0
    new_alerts:           list  = field(default_factory=list)
    break_reminder:       bool  = False

# ── Engine ────────────────────────────────────────────────────────────────────

class DetectionEngine:
    """
    Encapsulates camera, landmarks, calibration, adaptation, logging, and clip recording.
    """
    def __init__(
        self,
        driver_id:      int,
        driver_name:    str,
        age_group:      str,
        session_id:     Optional[int],
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.driver_id    = driver_id
        self.driver_name  = driver_name
        self.age_group    = age_group
        self.session_id   = session_id
        self._alert_cb    = alert_callback

        self._cap: Optional[cv2.VideoCapture] = None
        self._dlib_detector  = None
        self._dlib_predictor = None
        self._lStart = self._lEnd = 0
        self._rStart = self._rEnd = 0
        self._mStart = self._mEnd = 0

        self._drive_start_time    = time.time()
        self._total_paused_secs   = 0.0
        self._pause_start_time: Optional[float] = None

        self._paused  = False
        self._stopped = False

        self._calibrating          = True
        self._ear_samples: list    = []
        self._mar_samples: list    = []
        self._calibration_end_time = 0.0

        self._personal_baseline_ear = 0.25
        self._personal_baseline_mar = 0.65
        self._EYE_AR_THRESH         = 0.24
        self._MOU_AR_THRESH         = self._personal_baseline_mar
        self._HEAD_PITCH_THRESH     = 12
        self._DISTRACTION_YAW_THRESH= 20
        self._last_threshold_update = -ADAPTIVE_THRESH_INTERVAL

        self._eye_closed_start: Optional[float] = None
        self._eye_event_logged = False

        self._yawn_start: Optional[float] = None
        self._yawn_counted   = False
        self._yawn_status    = False
        self.yawn_timestamps: list[float] = []

        self._head_down_start: Optional[float]  = None
        self._distracted_start: Optional[float] = None

        self._no_face_counter      = 0
        self._consecutive_failures = 0

        self._last_head_down_log   = 0.0
        self._last_distracted_log  = 0.0
        self._break_reminder_sent  = False

        self._logger = SessionLogger(age_group)
        if session_id is not None:
            self._logger.set_session_id(session_id)

        self._clip_recorder = ClipRecorder()
        try:
            cleanup_old_clips()
        except Exception as e:
            _log.warning("cleanup_old_clips failed: %s", e)

        _log.info(
            "DetectionEngine initialised | driver=%s age=%s session=%s",
            driver_name, age_group, session_id,
        )

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    def open_camera(self) -> cv2.VideoCapture:
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            msg = f"'{model_path}' not found. Run: python download_model.py"
            _log.error(msg)
            raise FileNotFoundError(msg)

        _log.info("Loading Dlib facial landmark predictor…")
        self._dlib_detector  = dlib.get_frontal_face_detector()
        self._dlib_predictor = dlib.shape_predictor(model_path)
        self._lStart, self._lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self._rStart, self._rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self._mStart, self._mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        _log.info("Starting video stream…")
        for idx, backend, label in _CAMERA_ATTEMPTS:
            _log.info("Trying camera %s…", label)
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
                    ret_t, frm_t = cap.read()
                    if ret_t and frm_t is not None and frm_t.size > 0 and frm_t.mean() > 1.0:
                        success_count += 1

                if success_count >= 5:
                    _log.info("Camera opened successfully — %s", label)
                    self._cap = cap
                    self._drive_start_time = time.time()
                    return cap

                cap.release()
            except Exception as exc:
                _log.warning("%s — exception: %s", label, exc)

        _log.error("Could not open camera with any backend or index.")
        sys.exit(1)

    @staticmethod
    def _read_frame_safe(cap: cv2.VideoCapture, retries: int = 3):
        for _ in range(retries):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
            time.sleep(0.05)
        return False, None

    def pause(self):
        if not self._paused:
            self._paused = True
            self._pause_start_time = time.time()
            _log.info("Detection paused.")

    def resume(self):
        if self._paused:
            if self._pause_start_time is not None:
                self._total_paused_secs += time.time() - self._pause_start_time
                self._pause_start_time = None
            self._paused = False
            self._eye_closed_start = None
            self._eye_event_logged = False
            self._yawn_start       = None
            self._yawn_counted     = False
            self._head_down_start  = None
            self._distracted_start = None
            _log.info("Detection resumed.")

    def stop(self):
        self._stopped = True
        _log.info("Detection stop requested.")

    def get_active_drive_seconds(self) -> float:
        elapsed = time.time() - self._drive_start_time - self._total_paused_secs
        if self._paused and self._pause_start_time is not None:
            elapsed -= (time.time() - self._pause_start_time)
        return max(0.0, elapsed)

    def _fire_alert(self, alert_type: str, description: str, result: FrameResult):
        result.new_alerts.append({"type": alert_type, "desc": description})
        if self._alert_cb is not None:
            try:
                self._alert_cb(alert_type, description)
            except Exception:
                pass

    def process_frame(self) -> Optional[FrameResult]:
        if self._cap is None:
            raise RuntimeError("open_camera() must be called before process_frame()")

        ret, frame = self._read_frame_safe(self._cap)
        if not ret:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _log.error("Too many consecutive camera read failures. Stopping.")
                return None
            return FrameResult(frame=np.zeros((480, 640, 3), dtype=np.uint8))

        self._consecutive_failures = 0
        frame = imutils.resize(frame, width=640)
        frame_height, frame_width = frame.shape[:2]
        current_time = time.time()

        current_drive_seconds = self.get_active_drive_seconds()
        drive_minutes = current_drive_seconds / 60

        result = FrameResult(
            frame          = frame,
            ear_thresh     = self._EYE_AR_THRESH,
            mar_thresh     = self._MOU_AR_THRESH,
            calibrating    = self._calibrating,
            drive_seconds  = current_drive_seconds,
            yawns          = len(self.yawn_timestamps),
        )

        if self._paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            cv2.putText(frame, "PAUSED",
                        (frame_width // 2 - 80, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            result.frame      = frame
            result.status_str = "PAUSED"
            result.eye_status = "N/A"
            result.mouth_status = "N/A"
            self._eye_closed_start = None
            self._yawn_start       = None
            self._head_down_start  = None
            self._distracted_start = None
            return result

        effective_drive_minutes = max(0.0, (current_drive_seconds - CALIBRATION_DURATION) / 60)
        if effective_drive_minutes >= BREAK_REMINDER_MINUTES and not self._break_reminder_sent:
            self._break_reminder_sent = True
            result.break_reminder = True
            _log.warning("⚠ 2hr effective drive — consider a break")
            self._fire_alert("system", "⚠ 2hr drive — consider a break", result)

        if self._calibrating:
            remaining = int(CALIBRATION_DURATION - current_drive_seconds)
            result.calib_remaining = max(0, remaining)

            if remaining > 0:
                result.status_str = "CALIBRATING"
                cv2.putText(
                    frame, "CALIBRATING — Sit normally, keep eyes open",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
                cv2.putText(
                    frame, f"Time left: {remaining // 60}:{remaining % 60:02d}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
            else:
                self._finalize_calibration(current_time)
                result.calibration_complete = True
                result.calibrating = False

        if not self._calibrating and current_time - self._calibration_end_time < 5:
            cv2.putText(
                frame,
                f"Calibration Complete!  EAR Thresh: {self._EYE_AR_THRESH:.3f}",
                (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )

        if (not self._calibrating and
                current_drive_seconds - self._last_threshold_update >= ADAPTIVE_THRESH_INTERVAL):
            self._EYE_AR_THRESH = compute_adaptive_threshold(
                self._personal_baseline_ear, self.age_group, drive_minutes)
            self._MOU_AR_THRESH = compute_adaptive_mar_threshold(
                self._personal_baseline_mar, self.age_group, drive_minutes)
            self._last_threshold_update = current_drive_seconds
            _log.info(
                "Adaptive thresholds updated — EAR: %.3f  MAR: %.3f",
                self._EYE_AR_THRESH, self._MOU_AR_THRESH,
            )
        result.ear_thresh = self._EYE_AR_THRESH
        result.mar_thresh = self._MOU_AR_THRESH

        hours   = int(drive_minutes // 60)
        minutes = int(drive_minutes % 60)
        secs    = int(current_drive_seconds % 60)
        cv2.putText(frame, f"Drive Time: {hours:02d}:{minutes:02d}:{secs:02d}",
                    (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "P: Pause | Q: Quit",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, f"EAR Thresh: {self._EYE_AR_THRESH:.3f}",
                    (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not self._calibrating:
            completed = self._clip_recorder.add_frame(frame)
            if completed:
                _log.info("Clip saved: %s", completed)
        else:
            try:
                small = cv2.resize(frame, (480, 360))
                self._clip_recorder.buffer.append(small)
            except Exception:
                pass

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._dlib_detector(gray, 0)

        if len(rects) == 0:
            self._no_face_counter += 1
            result.face_detected = False
            if self._no_face_counter > NO_FACE_FRAME_LIMIT:
                cv2.putText(
                    frame, "FACE NOT DETECTED — Monitoring Paused",
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                )
                self._eye_closed_start = None
                self._eye_event_logged = False
                self._yawn_start       = None
                self._yawn_counted     = False
                self._yawn_status      = False
                self._head_down_start  = None
                self._distracted_start = None
            result.eye_status   = "N/A"
            result.mouth_status = "N/A"
        else:
            self._no_face_counter = 0

        for rect in rects:
            shape = self._dlib_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye  = shape[self._lStart:self._lEnd]
            right_eye = shape[self._rStart:self._rEnd]
            mouth     = shape[self._mStart:self._mEnd]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear       = (left_ear + right_ear) / 2.0
            mou_ear   = mouth_aspect_ratio(mouth)

            result.ear = ear
            result.mar = mou_ear

            # Collect calibration samples - filter out mid-blinks
            if self._calibrating:
                if ear > 0.20:
                    self._ear_samples.append(ear)
                self._mar_samples.append(mou_ear)

            reprojectdst, euler_angle = get_head_pose(shape, frame_width, frame_height)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            for start_pt, end_pt in line_pairs:
                cv2.line(frame,
                         tuple(map(int, reprojectdst[start_pt])),
                         tuple(map(int, reprojectdst[end_pt])),
                         (0, 0, 255))

            xx = euler_angle[0, 0]
            yy = euler_angle[1, 0]

            cv2.putText(frame, f"X: {xx:7.2f}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(frame, f"Y: {yy:7.2f}", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            if xx > self._HEAD_PITCH_THRESH:
                cv2.putText(frame, "HEAD DOWN",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not self._calibrating:
                    if self._head_down_start is None:
                        self._head_down_start = current_time
                    hd_elapsed = current_time - self._head_down_start
                    cv2.putText(frame, f"Head down: {hd_elapsed:.1f}s / {HEAD_DOWN_DURATION_THRESH:.1f}s",
                                (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    if hd_elapsed >= HEAD_DOWN_DURATION_THRESH:
                        sound_head_down_alarm()
                        if current_time - self._last_head_down_log >= LOG_COOLDOWN:
                            self._logger.log_event(
                                "head_down", drive_minutes, xx, self._HEAD_PITCH_THRESH)
                            self._fire_alert("head_down", "Head down", result)
                            self._last_head_down_log = current_time
            else:
                self._head_down_start = None

            if abs(yy) > self._DISTRACTION_YAW_THRESH:
                cv2.putText(frame, "DISTRACTED",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if not self._calibrating:
                    if self._distracted_start is None:
                        self._distracted_start = current_time
                    dist_elapsed = current_time - self._distracted_start
                    cv2.putText(frame, f"Distracted: {dist_elapsed:.1f}s / {DISTRACTED_DURATION_THRESH:.1f}s",
                                (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    if dist_elapsed >= DISTRACTED_DURATION_THRESH:
                        sound_distracted_alarm()
                        if current_time - self._last_distracted_log >= LOG_COOLDOWN:
                            self._logger.log_event(
                                "distracted", drive_minutes, yy, self._DISTRACTION_YAW_THRESH)
                            self._fire_alert("distracted", "Distracted", result)
                            self._last_distracted_log = current_time
            else:
                self._distracted_start = None

            if ear < self._EYE_AR_THRESH:
                result.eye_status = "Closed"
                cv2.putText(frame, "Eyes Closed",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not self._calibrating:
                    if self._eye_closed_start is None:
                        self._eye_closed_start = current_time

                    elapsed = current_time - self._eye_closed_start
                    cv2.putText(
                        frame,
                        f"Eye closed: {elapsed:.1f}s / {EYE_CLOSED_DURATION_THRESH:.1f}s",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                    )

                    if elapsed >= EYE_CLOSED_DURATION_THRESH:
                        result.eye_status = "ALERT"
                        cv2.putText(frame, "DROWSINESS ALERT!",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                        sound_eyes_closed_alarm()

                        clip_path = None
                        if not self._clip_recorder.is_recording():
                            try:
                                clip_path = self._clip_recorder.start_recording()
                            except Exception as ce:
                                _log.warning("Clip recording failed: %s", ce)

                        if not self._eye_event_logged:
                            self._logger.log_event(
                                "drowsiness_alarm", drive_minutes, ear,
                                self._EYE_AR_THRESH, clip_path=clip_path,
                            )
                            self._fire_alert("drowsiness_alarm", "Drowsiness detected", result)
                            self._eye_event_logged = True
            else:
                self._eye_closed_start = None
                self._eye_event_logged = False
                result.eye_status = "Open"
                cv2.putText(frame, "Eyes Open",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR: {ear:.3f}",
                        (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(frame, f"T:   {self._EYE_AR_THRESH:.3f}",
                        (480, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # ── Yawn (Sliding Window logic) ───────────────────────────────
            if mou_ear > self._MOU_AR_THRESH:
                result.mouth_status = "Yawning..."
                cv2.putText(frame, "Yawning...",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self._yawn_start is None:
                    self._yawn_start = current_time
                yawn_elapsed = current_time - self._yawn_start
                cv2.putText(frame, f"Yawn: {yawn_elapsed:.1f}s",
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                if yawn_elapsed >= YAWN_DURATION_THRESH:
                    self._yawn_status  = True
                    self._yawn_counted = True
            else:
                result.mouth_status = "Closed"
                if self._yawn_status:
                    self.yawn_timestamps.append(current_time)
                    self._yawn_status = False
                self._yawn_start   = None
                self._yawn_counted = False

            # Purge yawns outside 10 minutes
            self.yawn_timestamps = [t for t in self.yawn_timestamps if current_time - t <= YAWN_WINDOW_SECS]
            result.yawns = len(self.yawn_timestamps)

            if len(self.yawn_timestamps) > 0:
                cv2.putText(frame, f"Yawns: {len(self.yawn_timestamps)}/3",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)

            if len(self.yawn_timestamps) >= 3 and not self._calibrating:
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                sound_yawning_alarm()
                self._logger.log_event(
                    "yawn_detected", drive_minutes, mou_ear, self._MOU_AR_THRESH)
                self._fire_alert("yawn_detected", "3 yawns detected", result)
                self.yawn_timestamps.clear()

            cv2.putText(frame, f"MAR: {mou_ear:.3f}",
                        (480, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        if not self._calibrating:
            has_alert = any(
                a["type"] not in ("system",)
                for a in result.new_alerts
            ) or result.eye_status == "ALERT"
            result.status_str = "ALERT" if has_alert else "ACTIVE"
        else:
            result.status_str = "CALIBRATING"

        result.frame = frame
        return result

    def _finalize_calibration(self, current_time: float):
        self._calibrating         = False
        self._calibration_end_time= current_time

        if self._ear_samples:
            self._ear_samples.sort(reverse=True)
            slice_idx = max(1, int(len(self._ear_samples) * 0.7))
            top_70 = self._ear_samples[:slice_idx]
            if top_70:
                open_eye_avg = sum(top_70) / len(top_70)
                self._personal_baseline_ear = max(0.21, open_eye_avg - 0.06)
                self._EYE_AR_THRESH         = self._personal_baseline_ear
                self._logger.update_baseline(self._personal_baseline_ear)

        if self._mar_samples:
            self._mar_samples.sort()
            slice_idx = max(1, int(len(self._mar_samples) * 0.7))
            bot_70 = self._mar_samples[:slice_idx]
            if bot_70:
                closed_mouth_avg = sum(bot_70) / len(bot_70)
                self._personal_baseline_mar = max(0.53, closed_mouth_avg + 0.35)
                self._MOU_AR_THRESH         = self._personal_baseline_mar
                self._logger.update_baseline_mar(self._personal_baseline_mar)

        self._last_threshold_update = -ADAPTIVE_THRESH_INTERVAL
        _log.info(
            "Calibration complete — EAR baseline: %.3f  MAR baseline: %.3f",
            self._personal_baseline_ear, self._personal_baseline_mar,
        )

    def finalize(self) -> dict:
        try:
            if self._clip_recorder.is_recording():
                self._clip_recorder.stop_recording()
        except Exception as e:
            _log.warning("Error stopping clip recorder: %s", e)

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

        if self._yawn_status and not self._calibrating:
            _log.info("Flushing 1 confirmed in-progress yawn at session end.")
            final_drive_mins = self.get_active_drive_seconds() / 60
            try:
                self._logger.log_event(
                    "yawn_detected", final_drive_mins, 0.0, self._MOU_AR_THRESH
                )
            except Exception as e:
                _log.warning("Could not log end-of-session yawn: %s", e)

        if self._paused and self._pause_start_time is not None:
            self._total_paused_secs += time.time() - self._pause_start_time

        total_drive_mins = round(
            (time.time() - self._drive_start_time - self._total_paused_secs) / 60, 2
        )

        risk_score = None
        if self.session_id is not None:
            try:
                from analytics import compute_risk_score
                risk_score = compute_risk_score(self.session_id)
                _log.info("Risk Score: %s / 100", risk_score)
            except Exception as e:
                _log.warning("Could not compute risk score: %s", e)

        self._logger.save_and_print_summary(
            total_drive_mins,
            session_id  = self.session_id,
            risk_score  = risk_score,
        )

        _log.info("Stream ended. Total drive time: %.2f min", total_drive_mins)
        return {
            "total_drive_mins": total_drive_mins,
            "risk_score":       risk_score,
        }
