"""
database.py — SQLite persistence layer for Driver Fatigue Detection System.
Uses Python's built-in sqlite3. Creates fatigue.db in the project root.
All public functions are wrapped in try/except — DB failures never crash
the detection loop.
"""

import sqlite3
import os
from datetime import datetime

# ──────────────────────────────────────────────
# DB file location
# ──────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fatigue.db")


def _get_conn():
    """Open and return a new sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ──────────────────────────────────────────────
# Schema creation
# ──────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist. Called once at app startup."""
    try:
        conn = _get_conn()
        cur = conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS drivers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                age_group   TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id         INTEGER NOT NULL REFERENCES drivers(id),
                start_time        TEXT    NOT NULL,
                end_time          TEXT,
                total_drive_mins  REAL,
                baseline_ear      REAL,
                baseline_mar      REAL,
                risk_score        REAL,
                age_group         TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES sessions(id),
                event_type      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                drive_minute    REAL,
                metric_value    REAL,
                threshold_value REAL,
                clip_path       TEXT
            );
        """)

        conn.commit()
        conn.close()
        print("[DB] Database initialised →", _DB_PATH)
    except Exception as e:
        print(f"[DB WARNING] init_db failed: {e}")


# ──────────────────────────────────────────────
# Driver CRUD
# ──────────────────────────────────────────────
def create_driver(name: str, age_group: str) -> int | None:
    """Insert a new driver row. Returns the new driver id, or None on error."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO drivers (name, age_group, created_at) VALUES (?, ?, ?)",
            (name.strip(), age_group, datetime.now().isoformat()),
        )
        conn.commit()
        driver_id = cur.lastrowid
        conn.close()
        return driver_id
    except Exception as e:
        print(f"[DB WARNING] create_driver failed: {e}")
        return None


def get_all_drivers() -> list[dict]:
    """Return all drivers as a list of dicts (id, name, age_group, created_at)."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, name, age_group, created_at FROM drivers ORDER BY name")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB WARNING] get_all_drivers failed: {e}")
        return []


def get_driver_by_id(driver_id: int) -> dict | None:
    """Return a single driver dict or None if not found."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, age_group, created_at FROM drivers WHERE id = ?",
            (driver_id,),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[DB WARNING] get_driver_by_id failed: {e}")
        return None


# ──────────────────────────────────────────────
# Session CRUD
# ──────────────────────────────────────────────
def create_session(driver_id: int, age_group: str, start_time: str) -> int | None:
    """Insert a new session row. Returns the new session id, or None on error."""
    # Fix 10 — self-defending guard: driver_id=-1 is used in No-DB mode;
    # attempting INSERT with an invalid FK would raise a constraint error.
    if driver_id < 0:
        print(f"[DB] create_session skipped — No-DB mode (driver_id={driver_id})")
        return None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO sessions (driver_id, age_group, start_time)
               VALUES (?, ?, ?)""",
            (driver_id, age_group, start_time),
        )
        conn.commit()
        session_id = cur.lastrowid
        conn.close()
        return session_id
    except Exception as e:
        print(f"[DB WARNING] create_session failed: {e}")
        return None


def close_session(
    session_id: int,
    end_time: str,
    total_drive_mins: float,
    baseline_ear: float,
    baseline_mar: float,
    risk_score: float,
):
    """Update a session row when the drive ends."""
    try:
        conn = _get_conn()
        conn.execute(
            """UPDATE sessions
               SET end_time = ?, total_drive_mins = ?,
                   baseline_ear = ?, baseline_mar = ?, risk_score = ?
               WHERE id = ?""",
            (end_time, total_drive_mins, baseline_ear, baseline_mar, risk_score, session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB WARNING] close_session failed: {e}")


def get_sessions_for_driver(driver_id: int) -> list[dict]:
    """Return all sessions for a driver, newest first."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM sessions
               WHERE driver_id = ?
               ORDER BY start_time DESC""",
            (driver_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB WARNING] get_sessions_for_driver failed: {e}")
        return []


# ──────────────────────────────────────────────
# Event logging
# ──────────────────────────────────────────────
def log_event(
    session_id: int,
    event_type: str,
    timestamp: str,
    drive_minute: float,
    metric_value: float,
    threshold_value: float,
    clip_path: str | None = None,
):
    """Insert one event row."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO events
               (session_id, event_type, timestamp, drive_minute,
                metric_value, threshold_value, clip_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, event_type, timestamp, drive_minute,
             metric_value, threshold_value, clip_path),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB WARNING] log_event failed: {e}")


def get_events_for_session(session_id: int) -> list[dict]:
    """Return all events for a session."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY drive_minute",
            (session_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB WARNING] get_events_for_session failed: {e}")
        return []


# ──────────────────────────────────────────────
# Aggregated driver stats
# ──────────────────────────────────────────────
def get_driver_stats(driver_id: int) -> dict:
    """Return aggregated lifetime stats for a driver."""
    default = {
        "total_sessions": 0,
        "total_drive_mins": 0.0,
        "total_drowsy_alarms": 0,
        "total_yawn_alerts": 0,
        "total_head_downs": 0,
        "total_distractions": 0,
        "avg_risk_score": 0.0,
    }
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Session-level stats
        cur.execute(
            """SELECT COUNT(*) as cnt,
                      COALESCE(SUM(total_drive_mins), 0) as total_mins,
                      COALESCE(AVG(risk_score), 0) as avg_risk
               FROM sessions WHERE driver_id = ?""",
            (driver_id,),
        )
        row = cur.fetchone()
        default["total_sessions"]    = row["cnt"]
        default["total_drive_mins"]  = round(row["total_mins"] or 0.0, 2)
        default["avg_risk_score"]    = round(row["avg_risk"]   or 0.0, 1)

        # Event-level stats (across all sessions for this driver)
        event_types = {
            "drowsiness_alarm": "total_drowsy_alarms",
            "yawn_detected":    "total_yawn_alerts",
            "head_down":        "total_head_downs",
            "distracted":       "total_distractions",
        }
        for etype, key in event_types.items():
            cur.execute(
                """SELECT COUNT(*) as cnt FROM events e
                   JOIN sessions s ON s.id = e.session_id
                   WHERE s.driver_id = ? AND e.event_type = ?""",
                (driver_id, etype),
            )
            default[key] = cur.fetchone()["cnt"]

        conn.close()
        return default
    except Exception as e:
        print(f"[DB WARNING] get_driver_stats failed: {e}")
        return default
