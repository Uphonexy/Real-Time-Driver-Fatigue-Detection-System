"""
logger.py — In-memory session event logger for WakeMate v2.0.

v2.0 change (Phase 3 — DB-Primary strategy):
  save_and_print_summary() no longer writes a JSON file automatically.
  SQLite is now the single source of truth.  Use exporter.py to produce
  JSON or CSV files on demand from the History tab.

Existing API surface is preserved so detection_engine.py and the
deprecated main.py both work without changes.
"""

from datetime import datetime
try:
    from app_logger import get_logger as _get_logger
    _log = _get_logger("session_logger")
except ImportError:
    import logging
    _log = logging.getLogger("session_logger")

# DB import is optional — guarded so logger.py can still be imported
# even if database.py is not present.
try:
    import database as _db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


class SessionLogger:
    def __init__(self, age_group: str):
        self.session_log  = []
        self.session_meta = {
            "age_group":             age_group,
            "drive_start_time":      datetime.now().isoformat(),
            "personal_baseline_ear": None,
            "personal_baseline_mar": None,
        }
        self._session_id: int | None = None
        self._db_enabled: bool       = True

    # ── Baseline updates ──────────────────────────────────────────────────────

    def update_baseline(self, baseline: float):
        self.session_meta["personal_baseline_ear"] = baseline

    def update_baseline_mar(self, baseline_mar: float):
        self.session_meta["personal_baseline_mar"] = baseline_mar

    # ── DB integration ────────────────────────────────────────────────────────

    def set_session_id(self, session_id: int):
        self._session_id = session_id

    def set_db_enabled(self, flag: bool):
        """If False, DB operations are silently skipped."""
        self._db_enabled = flag

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type:        str,
        drive_minutes:     float,
        metric_value:      float,
        threshold_at_event: float,
        clip_path:         "str | None" = None,
    ):
        """
        Append an event to the in-memory log and persist it to the DB.

        clip_path is optional — only meaningful for drowsiness_alarm events.
        """
        # ── In-memory log ────────────────────────────────────────────────
        self.session_log.append({
            "event_type":         event_type,
            "timestamp":          datetime.now().isoformat(),
            "drive_minutes":      round(drive_minutes, 2),
            "metric_value":       round(float(metric_value), 3),
            "threshold_at_event": round(float(threshold_at_event), 3),
            **( {"clip_path": clip_path} if clip_path else {} ),
        })

        # ── DB log ────────────────────────────────────────────────────────
        if self._db_enabled and _DB_AVAILABLE and self._session_id is not None:
            try:
                _db.log_event(
                    session_id      = self._session_id,
                    event_type      = event_type,
                    timestamp       = datetime.now().isoformat(),
                    drive_minute    = round(drive_minutes, 2),
                    metric_value    = round(float(metric_value), 3),
                    threshold_value = round(float(threshold_at_event), 3),
                    clip_path       = clip_path,
                )
            except Exception as e:
                _log.warning("DB log_event failed: %s", e)

    # ── Session summary ───────────────────────────────────────────────────────

    def save_and_print_summary(
        self,
        total_drive_time:  float,
        session_id:        "int | None" = None,
        risk_score:        "float | None" = None,
    ):
        """
        Print the drive summary and close the DB session row.

        v2.0: JSON file is NO LONGER written automatically.
        Use exporter.export_session_json() / export_session_csv() on demand.
        """
        eyes_closed_count = sum(1 for e in self.session_log if e["event_type"] == "eyes_closed")
        drowsy_alarms     = sum(1 for e in self.session_log if e["event_type"] == "drowsiness_alarm")
        yawn_alerts       = sum(1 for e in self.session_log if e["event_type"] == "yawn_detected")
        head_downs        = sum(1 for e in self.session_log if e["event_type"] == "head_down")
        distracted        = sum(1 for e in self.session_log if e["event_type"] == "distracted")

        _log.info("=" * 40)
        _log.info("     DRIVER SESSION SUMMARY")
        _log.info("=" * 40)
        _log.info("Age Group:             %s",  self.session_meta["age_group"])
        _log.info("Total Drive Time:      %.2f mins", total_drive_time)
        _log.info("Baseline EAR:          %s",  self.session_meta["personal_baseline_ear"])
        _log.info("Baseline MAR:          %s",  self.session_meta["personal_baseline_mar"])
        _log.info("--- Events ---")
        _log.info("Eyes Closed Events:    %d",  eyes_closed_count)
        _log.info("Head Down Alerts:      %d",  head_downs)
        _log.info("Distracted Alerts:     %d",  distracted)
        _log.info("Yawn Cycle Alerts:     %d",  yawn_alerts)
        _log.info("Total Drowsy Alarms:   %d",  drowsy_alarms)
        if risk_score is not None:
            _log.info("Risk Score:            %.1f / 100", risk_score)
        _log.info("=" * 40)
        _log.info("Session data saved to SQLite (DB-primary mode). Use Export buttons for JSON/CSV.")

        # ── Close DB session ──────────────────────────────────────────────
        _sid = session_id if session_id is not None else self._session_id
        if self._db_enabled and _DB_AVAILABLE and _sid is not None:
            try:
                _db.close_session(
                    session_id       = _sid,
                    end_time         = datetime.now().isoformat(),
                    total_drive_mins = float(total_drive_time),
                    baseline_ear     = self.session_meta.get("personal_baseline_ear") or 0.0,
                    baseline_mar     = self.session_meta.get("personal_baseline_mar") or 0.0,
                    risk_score       = float(risk_score) if risk_score is not None else 0.0,
                )
            except Exception as e:
                _log.warning("DB close_session failed: %s", e)
