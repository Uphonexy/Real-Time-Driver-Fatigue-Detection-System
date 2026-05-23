"""
main.py — OpenCV/CLI entry point for WakeMate v2.0.

Runs the fatigue detection system in a plain OpenCV window (no Tkinter).
All detection logic has been extracted into DetectionEngine — this file
is now responsible only for:
  - Bootstrapping (DB init, driver setup)
  - The OpenCV display loop (imshow + waitKey)
  - Keyboard input (Q to quit, P to pause/resume)

Launch with:
    python main.py
"""

import sys
import time
from datetime import datetime

from pygame import mixer

from app_logger import get_logger
from database import init_db, create_session
from driver_manager import run_driver_setup, NO_DB_MODE
from detection_engine import DetectionEngine

import cv2

_log = get_logger("main")

# ── Audio init ────────────────────────────────────────────────────────────────
try:
    mixer.init()
except Exception as e:
    _log.warning("pygame mixer init failed: %s", e)


def main():
    # Fix 7 — guard for missing landmark model before anything else runs
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(
            f"[ERROR] '{model_path}' not found.\n"
            "Run: python download_model.py\n"
            "Then restart WakeMate."
        )
        sys.exit(1)

    # ── DB & driver setup ─────────────────────────────────────────────────────
    init_db()
    driver_id, driver_name, age_group = run_driver_setup()

    if NO_DB_MODE:
        _log.warning("Running in No-DB Mode — session data will NOT be saved.")

    session_id = None
    if driver_id != -1:
        session_id = create_session(driver_id, age_group, datetime.now().isoformat())

    _log.info(
        "Driver: %s | Age: %s | Session ID: %s",
        driver_name, age_group, session_id,
    )

    # ── Detection engine ──────────────────────────────────────────────────────
    engine = DetectionEngine(
        driver_id   = driver_id,
        driver_name = driver_name,
        age_group   = age_group,
        session_id  = session_id,
    )
    engine.open_camera()

    _log.info("Running… Press 'q' to quit, 'p' to pause/resume.")

    # ── Main display loop ─────────────────────────────────────────────────────
    try:
        while True:
            result = engine.process_frame()

            if result is None:
                _log.error("Camera failed — exiting.")
                break

            cv2.imshow("WakeMate — Driver Fatigue Detection", result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                if engine.is_paused:
                    engine.resume()
                    try:
                        mixer.music.stop()
                    except Exception:
                        pass
                else:
                    engine.pause()
                    try:
                        mixer.music.stop()
                    except Exception:
                        pass

    finally:
        cv2.destroyAllWindows()
        engine.finalize()


if __name__ == "__main__":
    main()