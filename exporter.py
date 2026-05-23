"""
exporter.py — On-demand session data export for WakeMate v2.0.

Replaces automatic JSON file creation (removed from logger.py in v2.0).
These functions are called by the History tab's Export buttons so researchers
or doctors can get offline copies of a session's data.

Functions
---------
export_session_json(session_id, output_path) -> str
    Write full session metadata + all events as JSON.

export_session_csv(session_id, output_path) -> str
    Write all events as a flat CSV (one row per event).
"""

import csv
import json
import os
from datetime import datetime
from typing import Optional

from app_logger import get_logger
from database import get_events_for_session

_log = get_logger("exporter")

try:
    import database as _db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


def _get_session_row(session_id: int) -> Optional[dict]:
    """Fetch raw session row from DB. Returns None if not found."""
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
    """
    Export a session's metadata and all events to a JSON file.

    Parameters
    ----------
    session_id  : DB session id
    output_path : Absolute path for the output file (including .json extension)

    Returns
    -------
    output_path on success.  Raises RuntimeError on failure.
    """
    session = _get_session_row(session_id)
    if session is None:
        raise RuntimeError(f"Session {session_id} not found in database.")

    events = get_events_for_session(session_id)

    payload = {
        "exported_at": datetime.now().isoformat(),
        "session_meta": {
            "session_id":       session.get("id"),
            "driver_id":        session.get("driver_id"),
            "age_group":        session.get("age_group"),
            "start_time":       session.get("start_time"),
            "end_time":         session.get("end_time"),
            "total_drive_mins": session.get("total_drive_mins"),
            "baseline_ear":     session.get("baseline_ear"),
            "baseline_mar":     session.get("baseline_mar"),
            "risk_score":       session.get("risk_score"),
        },
        "events": [
            {
                "event_type":      ev.get("event_type"),
                "timestamp":       ev.get("timestamp"),
                "drive_minute":    ev.get("drive_minute"),
                "metric_value":    ev.get("metric_value"),
                "threshold_value": ev.get("threshold_value"),
                "clip_path":       ev.get("clip_path"),
            }
            for ev in events
        ],
    }

    # Fix 5 — guard makedirs: dirname returns "" for filename-only paths
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(parent_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create output directory '{parent_dir}': {e}") from e

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    _log.info("Session %d exported to JSON: %s", session_id, output_path)
    return output_path


def export_session_csv(session_id: int, output_path: str) -> str:
    """
    Export all events for a session to a flat CSV file.

    Columns: session_id, event_type, timestamp, drive_minute,
             metric_value, threshold_value, clip_path

    Parameters
    ----------
    session_id  : DB session id
    output_path : Absolute path for the output file (including .csv extension)

    Returns
    -------
    output_path on success.  Raises RuntimeError on failure.
    """
    events = get_events_for_session(session_id)
    if events is None:
        raise RuntimeError(f"Could not retrieve events for session {session_id}.")

    fieldnames = [
        "session_id", "event_type", "timestamp", "drive_minute",
        "metric_value", "threshold_value", "clip_path",
    ]

    # Fix 5 — guard makedirs: dirname returns "" for filename-only paths
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(parent_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create output directory '{parent_dir}': {e}") from e

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            writer.writerow({
                "session_id":      session_id,
                "event_type":      ev.get("event_type", ""),
                "timestamp":       ev.get("timestamp", ""),
                "drive_minute":    ev.get("drive_minute", ""),
                "metric_value":    ev.get("metric_value", ""),
                "threshold_value": ev.get("threshold_value", ""),
                "clip_path":       ev.get("clip_path", ""),
            })

    _log.info("Session %d exported to CSV: %s", session_id, output_path)
    return output_path
