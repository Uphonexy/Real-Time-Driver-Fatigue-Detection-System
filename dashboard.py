"""
dashboard.py — Tkinter GUI entry point for WakeMate v2.0.

Runs the fatigue detection system with a full graphical dashboard.
Detection logic is entirely delegated to DetectionEngine — this file
is responsible only for:
  - The Tkinter UI (tabs, widgets, update loops)
  - Starting/stopping the background camera thread
  - Exporting session data (JSON / CSV) via exporter.py
  - Clip playback via the system default media player

Launch with:
    python dashboard.py
"""

__version__ = "2.0.0"

import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2  # Fix 2 — proper top-level import (was __import__("cv2") hack per frame)

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("\n[ERROR] Pillow library is required for the dashboard GUI.")
    print("Please install it using: pip install Pillow")
    sys.exit(1)

from pygame import mixer

from app_logger import get_logger
from database import init_db, create_session, get_sessions_for_driver, get_events_for_session
from driver_manager import run_driver_setup, NO_DB_MODE
from detection_engine import DetectionEngine, FrameResult
from analytics import compute_risk_score, get_risk_label
from exporter import export_session_json, export_session_csv

_log = get_logger("dashboard")

# ── Audio init ────────────────────────────────────────────────────────────────
try:
    mixer.init()
except Exception as e:
    _log.warning("pygame mixer init failed: %s", e)

# ── Shared state (written by camera thread, read by Tkinter thread) ───────────
state_lock   = threading.Lock()
shared_state = {
    "frame":                   None,
    "ear":                     0.0,
    "mar":                     0.0,
    "eye_status":              "Open",
    "mouth_status":            "Closed",
    "face_detected":           True,
    "yawns":                   0,
    "drive_seconds":           0.0,
    "calibrating":             True,
    "calib_remaining":         30,
    "paused":                  False,
    "stopped":                 False,
    "ear_thresh":              0.0,
    "mar_thresh":              0.0,
    "recent_alerts":           [],
    "status_str":              "CALIBRATING",
    "alert_active":            False,
    "calibration_complete_time": 0,
    "active_drive_seconds":    0.0,
    "break_reminder_shown":    False,
}


# ══════════════════════════════════════════════════════════════════════════════
# Background camera thread
# ══════════════════════════════════════════════════════════════════════════════

def camera_thread_func(driver_id, driver_name, age_group, session_id):
    """
    Run DetectionEngine in a background thread.  Writes results into
    shared_state under state_lock so the Tkinter UI can read them safely.
    """

    alert_clear_time = 0.0

    def add_alert(alert_type: str, description: str):
        """Push a new alert into shared_state (called from engine callback)."""
        nonlocal alert_clear_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        with state_lock:
            shared_state["recent_alerts"].insert(0, {
                "time": timestamp,
                "type": alert_type,
                "desc": description,
            })
            if len(shared_state["recent_alerts"]) > 5:
                shared_state["recent_alerts"].pop()
            shared_state["alert_active"] = True
        alert_clear_time = time.time() + 3.0

    engine = DetectionEngine(
        driver_id     = driver_id,
        driver_name   = driver_name,
        age_group     = age_group,
        session_id    = session_id,
        alert_callback= add_alert,
    )
    engine.open_camera()

    try:
        while True:
            with state_lock:
                is_stopped = shared_state["stopped"]
                is_paused  = shared_state["paused"]

            if is_stopped:
                break

            # Sync pause state into engine
            if is_paused and not engine.is_paused:
                engine.pause()
                try:
                    mixer.music.stop()
                except Exception:
                    pass
            elif not is_paused and engine.is_paused:
                engine.resume()

            result: FrameResult = engine.process_frame()

            if result is None:
                _log.error("Camera failed — stopping camera thread.")
                break

            # Expire alert banner
            if time.time() > alert_clear_time:
                with state_lock:
                    shared_state["alert_active"] = False

            # Alerts are already pushed instantly via the engine's alert_callback,
            # so we don't need to push them again here.

            # Break reminder — show once
            if result.break_reminder:
                with state_lock:
                    if not shared_state["break_reminder_shown"]:
                        shared_state["break_reminder_shown"] = True

            # Write state
            with state_lock:
                shared_state["frame"] = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                shared_state["ear"]                      = result.ear
                shared_state["mar"]                      = result.mar
                shared_state["eye_status"]               = result.eye_status
                shared_state["mouth_status"]             = result.mouth_status
                shared_state["face_detected"]            = result.face_detected
                shared_state["yawns"]                    = result.yawns
                shared_state["drive_seconds"]            = result.drive_seconds
                shared_state["active_drive_seconds"]     = result.drive_seconds
                shared_state["calib_remaining"]          = result.calib_remaining
                shared_state["ear_thresh"]               = result.ear_thresh
                shared_state["mar_thresh"]               = result.mar_thresh
                shared_state["status_str"]               = result.status_str
                if result.calibration_complete:
                    shared_state["calibrating"]          = False
                    shared_state["calibration_complete_time"] = time.time()

    finally:
        summary = engine.finalize()
        add_alert("system", f"Session ended — {summary.get('total_drive_mins', 0):.1f} min drive")
        _log.info("Camera thread finished.")


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard UI
# ══════════════════════════════════════════════════════════════════════════════

class DashboardApp:
    """Main Tkinter application window for WakeMate v2.0."""

    # ── Design tokens ──────────────────────────────────────────────────────────
    C_BG    = "#0a0f1e"
    C_PANEL = "#111827"
    C_CYAN  = "#00d4ff"
    C_RED   = "#ff3b3b"
    C_GREEN = "#00e676"
    C_AMBER = "#ffaa00"
    C_WHITE = "#ffffff"
    C_GRAY  = "#888888"

    def __init__(self, root: tk.Tk, driver_id, driver_name, age_group, session_id):
        self.root        = root
        self.driver_id   = driver_id
        self.driver_name = driver_name
        self.age_group   = age_group
        self.session_id  = session_id

        self.root.title(f"WAKEMATE v{__version__} — Driver Fatigue Detection")
        self.root.geometry("1150x710")
        self.root.configure(bg=self.C_BG)
        self.root.resizable(False, False)

        # App icon (graceful fallback if missing)
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            if os.path.exists(icon_path):
                icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon_img)
        except Exception as e:
            _log.warning("Could not set app icon: %s", e)

        # Flash state for blinking widgets
        self.flash_toggle = False

        # Treeview row-to-session-id mapping (for History export/play)
        self._history_session_map: dict[str, dict] = {}  # iid → session dict

        self._setup_ui()

        self.root.after(30,  self._update_feed)
        self.root.after(100, self._update_stats)
        self.root.after(500, self._toggle_flash)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ── Flash toggle ───────────────────────────────────────────────────────────
    def _toggle_flash(self):
        self.flash_toggle = not self.flash_toggle
        self.root.after(500, self._toggle_flash)

    # ── UI construction ────────────────────────────────────────────────────────
    def _setup_ui(self):
        # ── No-DB mode banner ────────────────────────────────────────────────
        if NO_DB_MODE:
            no_db_bar = tk.Frame(self.root, bg=self.C_AMBER, height=28)
            no_db_bar.pack(fill=tk.X, padx=0, pady=0)
            no_db_bar.pack_propagate(False)
            tk.Label(
                no_db_bar,
                text="⚠  Running in No-DB Mode — session data will NOT be saved.",
                font=("Segoe UI", 10, "bold"),
                bg=self.C_AMBER, fg="#000000",
            ).pack(expand=True)

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=self.C_BG)
        header.pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(
            header,
            text=f"WAKEMATE  🚗  Driver Fatigue Detection  v{__version__}",
            font=("Segoe UI", 18, "bold"),
            bg=self.C_BG, fg=self.C_WHITE,
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text=f"Driver: {self.driver_name}  |  Age: {self.age_group}",
            font=("Segoe UI", 11),
            bg=self.C_BG, fg=self.C_GRAY,
        ).pack(side=tk.RIGHT)

        # ── Tab bar ───────────────────────────────────────────────────────────
        self._tab_bar  = tk.Frame(self.root, bg=self.C_BG)
        self._tab_bar.pack(fill=tk.X, padx=20, pady=(8, 0))

        self._tab_content = tk.Frame(self.root, bg=self.C_BG)
        self._tab_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        self._tab_panels: dict[str, tk.Frame] = {}
        self._tab_btns:   dict[str, tk.Button] = {}

        self._build_live_panel()
        self._build_history_panel()
        self._show_tab("live")

        # ── Control panel ─────────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=self.C_PANEL)
        ctrl.pack(fill=tk.X, padx=20, pady=8)

        tk.Label(ctrl, text="CONTROL PANEL:", font=("Segoe UI", 12, "bold"),
                 bg=self.C_PANEL, fg=self.C_WHITE).pack(side=tk.LEFT, padx=10)

        self.btn_start = tk.Button(
            ctrl, text="▶ Start", font=("Segoe UI", 11, "bold"),
            bg=self.C_PANEL, fg=self.C_GREEN, bd=2, relief=tk.RAISED,
            command=self._cmd_start,
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = tk.Button(
            ctrl, text="⏸ Pause", font=("Segoe UI", 11, "bold"),
            bg=self.C_PANEL, fg=self.C_AMBER, bd=2, relief=tk.RAISED,
            command=self._cmd_pause,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            ctrl, text="⏹ Stop", font=("Segoe UI", 11, "bold"),
            bg=self.C_PANEL, fg=self.C_RED, bd=2, relief=tk.RAISED,
            command=self._cmd_stop,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        tk.Button(
            ctrl, text="🔄 Refresh History", font=("Segoe UI", 11),
            bg=self.C_PANEL, fg=self.C_CYAN, bd=1, relief=tk.FLAT,
            command=self._refresh_history,
        ).pack(side=tk.RIGHT, padx=10)

        tk.Label(
            self.root,
            text="Press buttons to control detection. Video streams continuously.",
            font=("Segoe UI", 10), bg=self.C_BG, fg=self.C_GRAY,
        ).pack(pady=0)

    # ── Tab helpers ────────────────────────────────────────────────────────────
    def _make_tab_btn(self, label: str, tab_name: str):
        btn = tk.Button(
            self._tab_bar, text=label,
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
            self._refresh_history()

    # ── Live monitor panel ─────────────────────────────────────────────────────
    def _build_live_panel(self):
        self._make_tab_btn("📷 Live Monitor", "live")
        panel = tk.Frame(self._tab_content, bg=self.C_BG)
        self._tab_panels["live"] = panel

        main_frame = tk.Frame(panel, bg=self.C_BG)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Column 1 — live feed
        col1 = tk.Frame(main_frame, bg=self.C_PANEL, width=500, height=450)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        col1.pack_propagate(False)

        # Column 2 — driver stats
        col2 = tk.Frame(main_frame, bg=self.C_PANEL, width=290, height=450)
        col2.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        col2.pack_propagate(False)

        # Column 3 — alerts
        col3 = tk.Frame(main_frame, bg=self.C_PANEL, width=290, height=450)
        col3.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        col3.pack_propagate(False)

        # ── Col 1 ─────────────────────────────────────────────────────────────
        tk.Label(col1, text="📷 Live Feed",
                 font=("Segoe UI", 14, "bold"), bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        self.feed_border = tk.Frame(col1, bg=self.C_PANEL, bd=3)
        self.feed_border.pack(pady=5)
        self.video_label = tk.Label(self.feed_border, bg="#000")
        self.video_label.pack()

        # ── Col 2 ─────────────────────────────────────────────────────────────
        tk.Label(col2, text="👤 Driver Stats",
                 font=("Segoe UI", 14, "bold"), bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        self.stats: dict[str, tk.Label] = {}
        rows = [
            ("Face detected", "✓ Active"),
            ("Eyes",          "Open"),
            ("Mouth",         "Closed"),
            ("EAR",           "0.000"),
            ("MAR",           "0.000"),
            ("EAR Threshold", "0.000"),
            ("Yawns",         "0 / 3"),
            ("Drive time",    "00:00:00"),
            ("Status",        "ACTIVE"),
        ]
        stats_frame = tk.Frame(col2, bg=self.C_PANEL)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        for name, default in rows:
            row = tk.Frame(stats_frame, bg=self.C_PANEL)
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=name,
                     font=("Segoe UI", 12), bg=self.C_PANEL, fg=self.C_WHITE).pack(side=tk.LEFT)
            lbl = tk.Label(row, text=default,
                           font=("Consolas", 12, "bold"), bg=self.C_PANEL, fg=self.C_CYAN)
            lbl.pack(side=tk.RIGHT)
            self.stats[name] = lbl

        self.lbl_calib_banner = tk.Label(col2, text="",
                                         font=("Segoe UI", 12, "bold"),
                                         bg=self.C_PANEL, fg=self.C_GREEN)
        self.lbl_calib_banner.pack(pady=5)

        # ── Col 3 ─────────────────────────────────────────────────────────────
        tk.Label(col3, text="⚠ Alerts (Last 5)",
                 font=("Segoe UI", 14, "bold"), bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=10)

        alerts_frame = tk.Frame(col3, bg=self.C_PANEL)
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.alert_widgets = []
        for _ in range(5):
            frm = tk.Frame(alerts_frame, bg=self.C_PANEL)
            frm.pack(fill=tk.X, pady=3)
            border = tk.Frame(frm, bg=self.C_PANEL, width=4)
            border.pack(side=tk.LEFT, fill=tk.Y)
            lbl = tk.Label(frm, text="",
                           font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_GRAY, anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, padx=5)
            self.alert_widgets.append((border, lbl))

        start_str = datetime.now().strftime("%H:%M:%S")
        tk.Label(col3, text=f"Session started: {start_str}",
                 font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_GRAY).pack(side=tk.BOTTOM, pady=10)

    # ── History panel ──────────────────────────────────────────────────────────
    def _build_history_panel(self):
        self._make_tab_btn("📊 History", "history")
        panel = tk.Frame(self._tab_content, bg=self.C_PANEL)
        self._tab_panels["history"] = panel

        tk.Label(panel, text="📊 Session History",
                 font=("Segoe UI", 15, "bold"), bg=self.C_PANEL, fg=self.C_CYAN).pack(pady=(12, 4))
        tk.Label(panel, text=f"Driver: {self.driver_name}  |  All past sessions",
                 font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_GRAY).pack(pady=(0, 8))

        # ── Treeview ──────────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("History.Treeview",
                        background=self.C_PANEL, foreground=self.C_WHITE,
                        fieldbackground=self.C_PANEL, rowheight=26,
                        font=("Segoe UI", 10))
        style.configure("History.Treeview.Heading",
                        background="#1e293b", foreground=self.C_CYAN,
                        font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("History.Treeview", background=[("selected", "#1e3a5f")])

        cols = ("Date", "Drive Time", "Risk Score", "Drowsy", "Yawns", "Head Down", "Distracted")
        tree_frame = tk.Frame(panel, bg=self.C_PANEL)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            style="History.Treeview", yscrollcommand=vsb.set,
        )
        vsb.config(command=self.history_tree.yview)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        col_widths = [120, 110, 130, 80, 80, 110, 110]
        for col, w in zip(cols, col_widths):
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=w, anchor="center")

        self.history_tree.tag_configure("low",      foreground="#00e676")
        self.history_tree.tag_configure("moderate", foreground="#ffaa00")
        self.history_tree.tag_configure("high",     foreground="#ff3b3b")
        self.history_tree.tag_configure("critical", foreground="#ff3b3b",
                                        font=("Segoe UI", 10, "bold"))

        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)
        self.history_tree.bind("<ButtonRelease-1>", self._on_history_select)

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = tk.Frame(panel, bg=self.C_PANEL)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.btn_export_json = tk.Button(
            btn_frame, text="⬇ Export JSON",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b", fg=self.C_CYAN, bd=1, relief=tk.GROOVE,
            state=tk.DISABLED,
            command=self._export_json,
        )
        self.btn_export_json.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_export_csv = tk.Button(
            btn_frame, text="⬇ Export CSV",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b", fg=self.C_CYAN, bd=1, relief=tk.GROOVE,
            state=tk.DISABLED,
            command=self._export_csv,
        )
        self.btn_export_csv.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_play_clip = tk.Button(
            btn_frame, text="▶ Play Clip",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b", fg=self.C_AMBER, bd=1, relief=tk.GROOVE,
            state=tk.DISABLED,
            command=self._play_clip,
        )
        self.btn_play_clip.pack(side=tk.LEFT, padx=(0, 6))

        # Store currently selected session
        self._selected_session: dict | None = None

    # ── History helpers ────────────────────────────────────────────────────────
    def _refresh_history(self):
        """Reload session history from DB into the Treeview."""
        try:
            for row in self.history_tree.get_children():
                self.history_tree.delete(row)
            self._history_session_map.clear()
            self._selected_session = None
            self._set_history_buttons(enabled=False)

            if NO_DB_MODE:
                self.history_tree.insert("", "end",
                                         values=("No-DB Mode — history unavailable",
                                                 "", "", "", "", "", ""))
                return

            sessions = get_sessions_for_driver(self.driver_id)
            if not sessions:
                self.history_tree.insert("", "end",
                                         values=("No sessions yet", "", "", "", "", "", ""))
                return

            for s in sessions:
                sid       = s.get("id", 0)
                start     = (s.get("start_time") or "")[:16].replace("T", "  ")
                drive_mins= s.get("total_drive_mins") or 0.0
                drive_str = f"{round(drive_mins, 1)} min"

                try:
                    evts = get_events_for_session(sid)
                    # Fix 6 — inline expressions replace the closure-inside-loop footgun
                    n_drowsy = sum(1 for e in evts if e.get("event_type") == "drowsiness_alarm")
                    n_yawn   = sum(1 for e in evts if e.get("event_type") == "yawn_detected")
                    n_head   = sum(1 for e in evts if e.get("event_type") == "head_down")
                    n_distr  = sum(1 for e in evts if e.get("event_type") == "distracted")
                except Exception:
                    n_drowsy = n_yawn = n_head = n_distr = "?"

                try:
                    score    = compute_risk_score(sid)
                    label    = get_risk_label(score)
                    risk_str = f"{score}  {label}"
                except Exception:
                    score    = 0.0
                    label    = "LOW RISK"
                    risk_str = "N/A"

                tag_map = {
                    "LOW RISK":      "low",
                    "MODERATE RISK": "moderate",
                    "HIGH RISK":     "high",
                    "CRITICAL RISK": "critical",
                }
                tag = tag_map.get(label, "low")

                iid = self.history_tree.insert(
                    "", "end",
                    values=(start, drive_str, risk_str, n_drowsy, n_yawn, n_head, n_distr),
                    tags=(tag,),
                )
                self._history_session_map[iid] = s

            # Auto-select the first session if available
            children = self.history_tree.get_children()
            if children and not NO_DB_MODE and sessions:
                self.history_tree.selection_set(children[0])
                self.history_tree.focus(children[0])
                self._on_history_select()

        except Exception as e:
            _log.warning("refresh_history failed: %s", e)

    def _on_history_select(self, _event=None):
        """Enable/disable action buttons based on whether a row is selected."""
        sel = self.history_tree.selection()
        if sel:
            iid = sel[0]
            self._selected_session = self._history_session_map.get(iid)
        else:
            self._selected_session = None
        self._set_history_buttons(enabled=bool(sel and self._selected_session))

    def _set_history_buttons(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_export_json.config(state=state)
        self.btn_export_csv.config(state=state)
        # Play clip only enabled if there are clip_paths in this session
        self.btn_play_clip.config(state=state)

    def _selected_session_id(self) -> int | None:
        if self._selected_session:
            return self._selected_session.get("id")
        return None

    def _export_json(self):
        sid = self._selected_session_id()
        if sid is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"session_{sid}.json",
            title="Export Session as JSON",
        )
        if not path:
            return
        try:
            export_session_json(sid, path)
            messagebox.showinfo("Export Complete", f"JSON saved to:\n{path}")
        except Exception as e:
            _log.error("JSON export failed: %s", e)
            messagebox.showerror("Export Failed", str(e))

    def _export_csv(self):
        sid = self._selected_session_id()
        if sid is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"session_{sid}.csv",
            title="Export Session as CSV",
        )
        if not path:
            return
        try:
            export_session_csv(sid, path)
            messagebox.showinfo("Export Complete", f"CSV saved to:\n{path}")
        except Exception as e:
            _log.error("CSV export failed: %s", e)
            messagebox.showerror("Export Failed", str(e))

    def _play_clip(self):
        """Find the most recent clip for this session and play it."""
        sid = self._selected_session_id()
        if sid is None:
            return

        try:
            evts = get_events_for_session(sid)
            clip_paths = [
                e.get("clip_path") for e in evts
                if e.get("clip_path") and os.path.exists(e.get("clip_path", ""))
            ]
        except Exception as e:
            _log.warning("Could not query clip paths: %s", e)
            clip_paths = []

        if not clip_paths:
            messagebox.showinfo("No Clip", "No video clip found for this session.")
            return

        clip = clip_paths[-1]  # most recent
        try:
            if sys.platform == "win32":
                os.startfile(clip)
            elif sys.platform == "darwin":
                subprocess.run(["open", clip])
            else:
                subprocess.run(["xdg-open", clip])
        except Exception as e:
            _log.error("Could not play clip %s: %s", clip, e)
            messagebox.showerror("Playback Error", f"Could not open clip:\n{clip}\n\n{e}")

    # ── Control buttons ────────────────────────────────────────────────────────
    def _cmd_start(self):
        with state_lock:
            shared_state["paused"] = False

    def _cmd_pause(self):
        with state_lock:
            shared_state["paused"] = True

    def _cmd_stop(self):
        with state_lock:
            shared_state["stopped"] = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        # Auto-refresh history after DB has had time to write
        self.root.after(900, self._refresh_history)

    def _on_closing(self):
        self._cmd_stop()
        self.root.after(500, self.root.destroy)

    def _show_break_reminder(self):
        """
        Fix 12 — Non-blocking break reminder.
        Uses a Toplevel window so the Tkinter event loop (and thus the video
        feed's after() callbacks) keeps running while the dialog is open.
        """
        win = tk.Toplevel(self.root)
        win.title("⚠ Break Reminder")
        win.geometry("380x160")
        win.configure(bg=self.C_PANEL)
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.focus_force()

        tk.Label(
            win, text="⚠  2-Hour Drive Alert",
            font=("Segoe UI", 15, "bold"), bg=self.C_PANEL, fg=self.C_AMBER,
        ).pack(pady=(18, 6))

        tk.Label(
            win,
            text="You have been driving for 2 hours.\nConsider taking a short break for your safety.",
            font=("Segoe UI", 10), bg=self.C_PANEL, fg=self.C_WHITE, justify="center",
        ).pack(pady=4)

        tk.Button(
            win, text="  OK — I'll take a break soon  ",
            font=("Segoe UI", 10, "bold"),
            bg=self.C_AMBER, fg="#000000", bd=0, padx=14, pady=8,
            command=win.destroy,
        ).pack(pady=(12, 0))

    # ── Update loops ───────────────────────────────────────────────────────────
    def _update_feed(self):
        with state_lock:
            frame_rgb      = shared_state["frame"]
            is_calibrating = shared_state["calibrating"]
            calib_rem      = shared_state["calib_remaining"]
            alert_active   = shared_state["alert_active"]
            break_shown    = shared_state["break_reminder_shown"]

        # Fix 12 — non-blocking break reminder.
        # messagebox.showwarning() was blocking the Tkinter event loop, freezing
        # the video feed until the user clicked OK. _show_break_reminder() opens
        # a Toplevel window instead, keeping after() callbacks running.
        if break_shown:
            with state_lock:
                shared_state["break_reminder_shown"] = False
            self._show_break_reminder()

        if frame_rgb is not None:
            img = Image.fromarray(frame_rgb)
            img = img.resize((480, 360), Image.Resampling.LANCZOS)

            if is_calibrating:
                draw = ImageDraw.Draw(img)
                draw.text((10, 330), f"CALIBRATING — {calib_rem}s left", fill=(0, 212, 255))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if alert_active:
            self.feed_border.config(bg=self.C_RED if self.flash_toggle else self.C_PANEL)
        elif is_calibrating:
            self.feed_border.config(bg=self.C_CYAN if self.flash_toggle else self.C_PANEL)
        else:
            self.feed_border.config(bg=self.C_PANEL)

        self.root.after(30, self._update_feed)

    def _update_stats(self):
        with state_lock:
            st = shared_state.copy()

        # Drive time display
        active_secs = st.get("active_drive_seconds", st["drive_seconds"])
        hrs  = int(active_secs // 3600)
        mins = int((active_secs % 3600) // 60)
        secs = int(active_secs % 60)
        dr_time = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        # Face detected
        if st["face_detected"]:
            self.stats["Face detected"].config(text="✓ Active", fg=self.C_GREEN)
        else:
            self.stats["Face detected"].config(
                text="✗ Not detected",
                fg=self.C_RED if self.flash_toggle else self.C_PANEL,
            )

        # Eyes
        eye_color = self.C_GREEN
        es = st["eye_status"]
        if es == "Closed":
            eye_color = self.C_RED
        elif es == "ALERT":
            eye_color = self.C_RED if self.flash_toggle else self.C_PANEL
        elif es == "N/A":
            eye_color = self.C_GRAY
        self.stats["Eyes"].config(text=es, fg=eye_color)

        # Mouth
        m_color = self.C_GREEN
        ms = st["mouth_status"]
        if ms == "Yawning...":
            m_color = self.C_AMBER if self.flash_toggle else self.C_PANEL
        elif ms == "N/A":
            m_color = self.C_GRAY
        self.stats["Mouth"].config(text=ms, fg=m_color)

        # EAR
        ear_color = self.C_CYAN
        if st["ear"] < st["ear_thresh"] and st["ear"] > 0:
            ear_color = self.C_RED
        self.stats["EAR"].config(text=f"{st['ear']:.3f}", fg=ear_color)

        # MAR
        mar_color = self.C_CYAN
        if st["mar"] > st["mar_thresh"] and st["mar"] > 0:
            mar_color = self.C_AMBER
        self.stats["MAR"].config(text=f"{st['mar']:.3f}", fg=mar_color)

        # EAR Threshold
        self.stats["EAR Threshold"].config(text=f"{st['ear_thresh']:.3f}", fg=self.C_WHITE)

        # Yawns
        y_color = self.C_WHITE
        if st["yawns"] > 0:
            y_color = self.C_AMBER
        if st["yawns"] >= 3:
            y_color = self.C_RED
        self.stats["Yawns"].config(text=f"{st['yawns']} / 3", fg=y_color)

        # Drive time
        self.stats["Drive time"].config(text=dr_time, fg=self.C_WHITE)

        # Status
        s_color = self.C_GREEN
        ss = st["status_str"]
        if "PAUSED" in ss:
            s_color = self.C_AMBER
        elif "CALIBRATING" in ss:
            s_color = self.C_CYAN if self.flash_toggle else self.C_PANEL
        elif "ALERT" in ss:
            s_color = self.C_RED if self.flash_toggle else self.C_PANEL
        self.stats["Status"].config(text=ss, fg=s_color)

        # Calibration complete banner
        if not st["calibrating"] and time.time() - st["calibration_complete_time"] < 3:
            self.lbl_calib_banner.config(text="✓ Calibration complete")
        else:
            self.lbl_calib_banner.config(text="")

        # Alerts panel
        alert_colors = {
            "drowsiness_alarm": self.C_RED,
            "yawn_detected":    self.C_AMBER,
            "head_down":        self.C_AMBER,
            "distracted":       self.C_CYAN,
            "system":           self.C_GREEN,
        }
        if not st["recent_alerts"]:
            self.alert_widgets[0][1].config(text="None", fg=self.C_GRAY)
            self.alert_widgets[0][0].config(bg=self.C_PANEL)
            for bw, lw in self.alert_widgets[1:]:
                lw.config(text="")
                bw.config(bg=self.C_PANEL)
        else:
            for i in range(5):
                if i < len(st["recent_alerts"]):
                    al = st["recent_alerts"][i]
                    c  = alert_colors.get(al["type"], self.C_WHITE)
                    self.alert_widgets[i][1].config(
                        text=f"{al['time']}  {al['desc']}", fg=self.C_WHITE)
                    self.alert_widgets[i][0].config(bg=c)
                else:
                    self.alert_widgets[i][1].config(text="")
                    self.alert_widgets[i][0].config(bg=self.C_PANEL)

        self.root.after(100, self._update_stats)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Enable High-DPI awareness on Windows for crisp GUI
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        root_check = tk.Tk()
        root_check.withdraw()
        messagebox.showerror(
            "Missing Model File",
            f"'{model_path}' not found.\n\n"
            "Please run:\n    python download_model.py\n\n"
            "Then restart the dashboard.",
        )
        root_check.destroy()
        sys.exit(1)

    init_db()

    driver_id, driver_name, age_group = run_driver_setup()
    if driver_id is None:
        _log.info("Driver setup aborted.")
        sys.exit(0)

    if NO_DB_MODE:
        _log.warning("Running in No-DB Mode — session data will NOT be saved.")

    session_id = None
    if driver_id != -1:
        session_id = create_session(driver_id, age_group, datetime.now().isoformat())

    _log.info(
        "Driver: %s | Age: %s | Session ID: %s",
        driver_name, age_group, session_id,
    )

    bg_thread = threading.Thread(
        target=camera_thread_func,
        args=(driver_id, driver_name, age_group, session_id),
        daemon=True,
    )
    bg_thread.start()

    root = tk.Tk()
    app  = DashboardApp(root, driver_id, driver_name, age_group, session_id)
    root.mainloop()


if __name__ == "__main__":
    main()
