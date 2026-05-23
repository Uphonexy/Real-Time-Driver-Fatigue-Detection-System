"""
exporter.py — On-demand session data export for WakeMate v2.0.
"""
import csv, json, os
from datetime import datetime
from app_logger import get_logger
from database import get_events_for_session

_log = get_logger("exporter")

try:
    import database as _db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

def _get_session_row(session_id: int) -> dict | None:
    if not _DB_AVAILABLE:
        return None
    try:
        conn = _db._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        _log.warning("_get_session_row failed: %s", e)
        return None

def export_session_json(session_id: int, output_path: str) -> str:
    session = _get_session_row(session_id)
    if session is None:
        raise RuntimeError(f"Session {session_id} not found in database.")
    events = get_events_for_session(session_id)
    payload = {
        "exported_at": datetime.now().isoformat(),
        "session_meta": {k: session.get(k) for k in
            ("id","driver_id","age_group","start_time","end_time",
             "total_drive_mins","baseline_ear","baseline_mar","risk_score")},
        "events": [
            {k: ev.get(k) for k in
             ("event_type","timestamp","drive_minute","metric_value",
              "threshold_value","clip_path")}
            for ev in events
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    _log.info("Session %d exported to JSON: %s", session_id, output_path)
    return output_path

def export_session_csv(session_id: int, output_path: str) -> str:
    events = get_events_for_session(session_id)
    if events is None:
        raise RuntimeError(f"Could not retrieve events for session {session_id}.")
    fieldnames = ["session_id","event_type","timestamp","drive_minute",
                  "metric_value","threshold_value","clip_path"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            writer.writerow({**{k: ev.get(k,"") for k in fieldnames},
                             "session_id": session_id})
    _log.info("Session %d exported to CSV: %s", session_id, output_path)
    return output_path
