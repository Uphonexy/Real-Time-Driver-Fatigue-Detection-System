import json
from datetime import datetime

# DB import is optional — guarded so logger.py can still be imported
# even if database.py is not present.
try:
    import database as _db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


class SessionLogger:
    def __init__(self, age_group):
        self.session_log = []
        self.session_meta = {
            "age_group": age_group,
            "drive_start_time": datetime.now().isoformat(),
            "personal_baseline_ear": None,
            "personal_baseline_mar": None  # FIX 5: MAR calibration from samples
        }
        # DB integration (disabled until set_session_id is called)
        self._session_id: int | None = None
        self._db_enabled: bool = True

    # ──────────────────────────────────────────
    # Existing methods — UNCHANGED
    # ──────────────────────────────────────────

    def update_baseline(self, baseline):
        self.session_meta["personal_baseline_ear"] = baseline

    # FIX 5: Update MAR baseline in logger
    def update_baseline_mar(self, baseline_mar):
        self.session_meta["personal_baseline_mar"] = baseline_mar

    # ──────────────────────────────────────────
    # New DB-integration methods
    # ──────────────────────────────────────────

    def set_session_id(self, session_id: int):
        """Store the DB session id so that DB calls can be made."""
        self._session_id = session_id

    def set_db_enabled(self, flag: bool):
        """
        If False, all DB operations are silently skipped.
        JSON logging is always performed regardless of this flag.
        """
        self._db_enabled = flag

    # ──────────────────────────────────────────
    # MODIFIED: log_event — adds optional clip_path
    # ──────────────────────────────────────────

    def log_event(self, event_type, drive_minutes, metric_value,
                  threshold_at_event, clip_path=None):
        """
        Log an event to the in-memory JSON log.
        Also persists to the database if DB is enabled and session_id is set.

        clip_path is optional — only meaningful for drowsiness_alarm events.
        """
        # ── JSON log (unchanged behaviour) ────
        self.session_log.append({
            "event_type":        event_type,
            "timestamp":         datetime.now().isoformat(),
            "drive_minutes":     round(drive_minutes, 2),
            "metric_value":      round(float(metric_value), 3),
            "threshold_at_event": round(float(threshold_at_event), 3),
            **({"clip_path": clip_path} if clip_path else {}),
        })

        # ── DB log ────────────────────────────
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
                print(f"[Logger WARNING] DB log_event failed: {e}")

    # ──────────────────────────────────────────
    # MODIFIED: save_and_print_summary — adds optional session_id & risk_score
    # ──────────────────────────────────────────

    def save_and_print_summary(self, total_drive_time,
                               session_id=None, risk_score=None):
        """
        Save the JSON session file and print the summary to stdout.
        Existing behaviour is entirely preserved.

        Additionally closes the DB session row when session_id is provided
        and DB is enabled.
        """
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_log = {
            "session_meta": self.session_meta,
            "events":       self.session_log,
        }

        with open(filename, 'w') as f:
            json.dump(full_log, f, indent=4)

        eyes_closed_count = sum(1 for e in self.session_log if e['event_type'] == 'eyes_closed')
        drowsy_alarms     = sum(1 for e in self.session_log if e['event_type'] == 'drowsiness_alarm')
        yawn_alerts       = sum(1 for e in self.session_log if e['event_type'] == 'yawn_detected')
        head_downs        = sum(1 for e in self.session_log if e['event_type'] == 'head_down')
        distracted        = sum(1 for e in self.session_log if e['event_type'] == 'distracted')

        print("\n" + "=" * 40)
        print("     DRIVER SESSION SUMMARY")
        print("=" * 40)
        print(f"Age Group:             {self.session_meta['age_group']}")
        print(f"Total Drive Time:      {total_drive_time} mins")
        print(f"Baseline EAR:          {self.session_meta['personal_baseline_ear']}")
        print(f"Baseline MAR:          {self.session_meta['personal_baseline_mar']}")  # FIX 5
        print(f"\n--- Events ---")
        print(f"Eyes Closed Events:    {eyes_closed_count}")
        print(f"Head Down Alerts:      {head_downs}")
        print(f"Distracted Alerts:     {distracted}")
        print(f"Yawn Cycle Alerts:     {yawn_alerts}")
        print(f"Total Drowsy Alarms:   {drowsy_alarms}")
        if risk_score is not None:
            print(f"Risk Score:            {risk_score} / 100")
        print("=" * 40)
        print(f"[INFO] Log saved to {filename}")

        # ── Close DB session ──────────────────
        _sid = session_id if session_id is not None else self._session_id
        if self._db_enabled and _DB_AVAILABLE and _sid is not None:
            try:
                _db.close_session(
                    session_id      = _sid,
                    end_time        = datetime.now().isoformat(),
                    total_drive_mins= float(total_drive_time),
                    baseline_ear    = self.session_meta.get("personal_baseline_ear") or 0.0,
                    baseline_mar    = self.session_meta.get("personal_baseline_mar") or 0.0,
                    risk_score      = float(risk_score) if risk_score is not None else 0.0,
                )
            except Exception as e:
                print(f"[Logger WARNING] DB close_session failed: {e}")
