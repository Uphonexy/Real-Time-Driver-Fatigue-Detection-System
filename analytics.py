"""
analytics.py — Pure analytics functions for the Driver Fatigue Detection System.

All functions query the database via database.py and return plain Python dicts
or lists. No state is kept here. All functions are safe to call even if
the DB is empty or the requested IDs do not exist.
"""

from database import (
    get_events_for_session,
    get_sessions_for_driver,
    get_driver_by_id,
)
import database as _db


# ──────────────────────────────────────────────
# Risk scoring
# ──────────────────────────────────────────────

# Weights and caps per event type
_WEIGHTS = {
    "drowsiness_alarm": (25, 100),
    "yawn_detected":    (10,  40),
    "head_down":        ( 5,  30),
    "distracted":       ( 5,  30),
}


def compute_risk_score(session_id: int) -> float:
    """
    Compute a risk score (0.0–100.0) for a session based on event counts.

    Weights:
        drowsiness_alarm : 25 pts each  (cap 100)
        yawn_detected    : 10 pts each  (cap  40)
        head_down        :  5 pts each  (cap  30)
        distracted       :  5 pts each  (cap  30)

    Final score = min(100, sum of capped weighted values).
    """
    try:
        events = get_events_for_session(session_id)
        if not events:
            return 0.0

        # Count each event type
        counts: dict[str, int] = {}
        for ev in events:
            etype = ev.get("event_type", "")
            counts[etype] = counts.get(etype, 0) + 1

        total = 0.0
        for etype, (pts, cap) in _WEIGHTS.items():
            contribution = min(counts.get(etype, 0) * pts, cap)
            total += contribution

        return round(min(100.0, total), 1)

    except Exception as e:
        print(f"[Analytics WARNING] compute_risk_score failed: {e}")
        return 0.0


def get_risk_label(score: float) -> str:
    """Map a numeric risk score to a human-readable label."""
    if score <= 25:
        return "LOW RISK"
    elif score <= 50:
        return "MODERATE RISK"
    elif score <= 75:
        return "HIGH RISK"
    else:
        return "CRITICAL RISK"


# ──────────────────────────────────────────────
# Session summary
# ──────────────────────────────────────────────

def get_session_summary(session_id: int) -> dict:
    """
    Return a comprehensive summary dict for a session:
        session_id, driver_name, age_group,
        start_time, end_time, total_drive_mins,
        baseline_ear, baseline_mar,
        risk_score, risk_label,
        event_counts  (dict of event_type → count),
        events_per_hour,
        most_dangerous_minute
    """
    default = {
        "session_id":           session_id,
        "driver_name":          "Unknown",
        "age_group":            "N/A",
        "start_time":           None,
        "end_time":             None,
        "total_drive_mins":     0.0,
        "baseline_ear":         None,
        "baseline_mar":         None,
        "risk_score":           0.0,
        "risk_label":           "LOW RISK",
        "event_counts":         {},
        "events_per_hour":      0.0,
        "most_dangerous_minute": None,
    }
    try:
        # Fetch session row directly
        conn = _db._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        sess_row = cur.fetchone()
        conn.close()

        if sess_row is None:
            return default

        sess = dict(sess_row)

        # Driver info
        driver = get_driver_by_id(sess.get("driver_id", 0)) or {}
        driver_name = driver.get("name", "Unknown")

        # Events
        events = get_events_for_session(session_id)
        event_counts: dict[str, int] = {}
        minute_freq:  dict[int, int]  = {}

        for ev in events:
            etype = ev.get("event_type", "other")
            event_counts[etype] = event_counts.get(etype, 0) + 1

            dm = ev.get("drive_minute")
            if dm is not None:
                bucket = int(dm)
                minute_freq[bucket] = minute_freq.get(bucket, 0) + 1

        # Risk
        risk_score = compute_risk_score(session_id)
        risk_label = get_risk_label(risk_score)

        # Events per hour
        drive_mins = sess.get("total_drive_mins") or 0.0
        total_events = len(events)
        events_per_hour = round(total_events / (drive_mins / 60), 2) if drive_mins > 0 else 0.0

        # Most dangerous minute
        most_dangerous_minute = (
            max(minute_freq, key=minute_freq.get) if minute_freq else None
        )

        return {
            "session_id":            session_id,
            "driver_name":           driver_name,
            "age_group":             sess.get("age_group", "N/A"),
            "start_time":            sess.get("start_time"),
            "end_time":              sess.get("end_time"),
            "total_drive_mins":      round(drive_mins, 2),
            "baseline_ear":          sess.get("baseline_ear"),
            "baseline_mar":          sess.get("baseline_mar"),
            "risk_score":            risk_score,
            "risk_label":            risk_label,
            "event_counts":          event_counts,
            "events_per_hour":       events_per_hour,
            "most_dangerous_minute": most_dangerous_minute,
        }

    except Exception as e:
        print(f"[Analytics WARNING] get_session_summary failed: {e}")
        return default


# ──────────────────────────────────────────────
# Driver trend (for charting)
# ──────────────────────────────────────────────

def get_driver_trend(driver_id: int) -> list[dict]:
    """
    Return the last 10 sessions for a driver (oldest→newest) with:
        session_id, date, drive_mins, risk_score, risk_label
    Suitable for plotting a risk trend over time.
    """
    try:
        sessions = get_sessions_for_driver(driver_id)  # newest first
        last_10 = sessions[:10]
        last_10.reverse()  # oldest first for charting

        result = []
        for s in last_10:
            sid   = s.get("id", 0)
            score = compute_risk_score(sid)
            result.append({
                "session_id": sid,
                "date":       (s.get("start_time") or "")[:10],
                "drive_mins": round(s.get("total_drive_mins") or 0.0, 2),
                "risk_score": score,
                "risk_label": get_risk_label(score),
            })
        return result

    except Exception as e:
        print(f"[Analytics WARNING] get_driver_trend failed: {e}")
        return []


# ──────────────────────────────────────────────
# Fatigue timeline (for plotting)
# ──────────────────────────────────────────────

def get_fatigue_timeline(session_id: int) -> list[dict]:
    """
    Return all events for a session sorted by drive_minute with:
        drive_minute, event_type, metric_value
    Useful for plotting when fatigue events occurred during the drive.
    """
    try:
        events = get_events_for_session(session_id)
        return [
            {
                "drive_minute": ev.get("drive_minute", 0.0),
                "event_type":   ev.get("event_type", ""),
                "metric_value": ev.get("metric_value", 0.0),
            }
            for ev in sorted(events, key=lambda e: e.get("drive_minute") or 0)
        ]
    except Exception as e:
        print(f"[Analytics WARNING] get_fatigue_timeline failed: {e}")
        return []
