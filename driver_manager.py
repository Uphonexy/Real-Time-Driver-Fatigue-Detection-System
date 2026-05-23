"""
driver_manager.py — Tkinter driver-setup dialog for the Fatigue Detection System.

Presents two modes:
  A) Select an existing driver from the database
  B) Create a new driver (name + age group)

Returns (driver_id, driver_name, age_group).
Calls sys.exit(0) if the window is closed without a selection.

v2.0: Adds NO_DB_MODE flag.  When the database fails and driver_id falls
back to -1, NO_DB_MODE is set to True so the DashboardApp can display a
persistent warning banner instead of silently failing later with FK errors.
"""

import sys
import tkinter as tk
from tkinter import messagebox
from database import get_all_drivers, create_driver
from app_logger import get_logger

_log = get_logger("driver_manager")

# ── Public flag — checked by dashboard.py to show No-DB banner ───────────────
NO_DB_MODE: bool = False

# ── Design tokens ─────────────────────────────────────────────────────────────
C_BG    = "#0a0f1e"
C_PANEL = "#111827"
C_CYAN  = "#00d4ff"
C_GREEN = "#00e676"
C_AMBER = "#ffaa00"
C_RED   = "#ff3b3b"
C_WHITE = "#ffffff"
C_GRAY  = "#888888"
C_INPUT = "#1e293b"

FONT_HEAD  = ("Segoe UI", 14, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_BTN   = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI",  9)

AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]


# ── Public entry point ────────────────────────────────────────────────────────

def run_driver_setup() -> tuple[int, str, str]:
    """
    Display the Driver Setup dialog and return (driver_id, driver_name, age_group).
    Blocks until the user makes a selection.  Calls sys.exit(0) on window close.

    If the DB fails and driver_id is -1, sets the module-level NO_DB_MODE = True.
    """
    global NO_DB_MODE

    result = [None]   # mutable container so inner callbacks can write to it

    root = tk.Tk()
    root.title("WAKEMATE — Driver Setup")
    root.geometry("400x350")
    root.configure(bg=C_BG)
    root.resizable(False, False)
    root.eval("tk::PlaceWindow . center")

    def on_close():
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # ── Header ───────────────────────────────────────────────────────────────
    tk.Label(
        root,
        text="🚗  WAKEMATE — Driver Setup",
        font=("Segoe UI", 15, "bold"),
        bg=C_BG, fg=C_CYAN,
    ).pack(pady=(18, 2))

    tk.Label(
        root, text="Identify the driver to begin monitoring.",
        font=FONT_SMALL, bg=C_BG, fg=C_GRAY,
    ).pack()

    # ── Tab switcher ──────────────────────────────────────────────────────────
    tab_frame = tk.Frame(root, bg=C_BG)
    tab_frame.pack(pady=(10, 0))

    content_frame = tk.Frame(root, bg=C_PANEL, bd=0, relief=tk.FLAT)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

    panels: dict[str, tk.Frame] = {}

    def show_panel(name: str):
        for panel in panels.values():
            panel.pack_forget()
        panels[name].pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        btn_existing.config(
            fg=C_CYAN if name == "existing" else C_GRAY,
            relief=tk.SUNKEN if name == "existing" else tk.FLAT,
        )
        btn_new.config(
            fg=C_CYAN if name == "new" else C_GRAY,
            relief=tk.SUNKEN if name == "new" else tk.FLAT,
        )

    btn_existing = tk.Button(
        tab_frame, text="Existing Driver",
        font=FONT_BTN, bg=C_PANEL, fg=C_CYAN, bd=1, relief=tk.SUNKEN,
        activebackground=C_PANEL, activeforeground=C_CYAN,
        command=lambda: show_panel("existing"),
    )
    btn_existing.pack(side=tk.LEFT, padx=4)

    btn_new = tk.Button(
        tab_frame, text="New Driver",
        font=FONT_BTN, bg=C_PANEL, fg=C_GRAY, bd=1, relief=tk.FLAT,
        activebackground=C_PANEL, activeforeground=C_CYAN,
        command=lambda: show_panel("new"),
    )
    btn_new.pack(side=tk.LEFT, padx=4)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL A — Existing Driver
    # ════════════════════════════════════════════════════════════════════════
    panel_existing = tk.Frame(content_frame, bg=C_PANEL)
    panels["existing"] = panel_existing

    tk.Label(panel_existing, text="Select Driver", font=FONT_HEAD,
             bg=C_PANEL, fg=C_CYAN).pack(anchor="w")
    tk.Label(panel_existing, text="Choose from previously registered drivers:",
             font=FONT_SMALL, bg=C_PANEL, fg=C_GRAY).pack(anchor="w", pady=(0, 8))

    list_frame = tk.Frame(panel_existing, bg=C_PANEL)
    list_frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    driver_listbox = tk.Listbox(
        list_frame,
        font=("Consolas", 11),
        bg=C_INPUT, fg=C_WHITE,
        selectbackground=C_CYAN, selectforeground="#000000",
        activestyle="none", bd=0, highlightthickness=1,
        highlightcolor=C_CYAN, highlightbackground=C_PANEL,
        yscrollcommand=scrollbar.set,
        height=6,
    )
    driver_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=driver_listbox.yview)

    all_drivers = get_all_drivers()
    driver_map: dict[int, dict] = {}

    if all_drivers:
        for i, drv in enumerate(all_drivers):
            driver_listbox.insert(tk.END, f"  {drv['name']}  ({drv['age_group']})")
            driver_map[i] = drv
    else:
        driver_listbox.insert(tk.END, "  — No drivers registered yet —")
        driver_listbox.config(state=tk.DISABLED)

    def cmd_start_existing():
        sel = driver_listbox.curselection()
        if not sel or not all_drivers:
            messagebox.showwarning("No Selection", "Please select a driver from the list.", parent=root)
            return
        drv = driver_map[sel[0]]
        result[0] = (drv["id"], drv["name"], drv["age_group"])
        root.destroy()

    tk.Button(
        panel_existing, text="▶  Start Session",
        font=FONT_BTN, bg=C_GREEN, fg="#000000", bd=0, relief=tk.FLAT,
        activebackground="#00c060", activeforeground="#000000",
        padx=12, pady=6,
        command=cmd_start_existing,
    ).pack(pady=(12, 0), anchor="e")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL B — New Driver
    # ════════════════════════════════════════════════════════════════════════
    panel_new = tk.Frame(content_frame, bg=C_PANEL)
    panels["new"] = panel_new

    tk.Label(panel_new, text="Create New Driver", font=FONT_HEAD,
             bg=C_PANEL, fg=C_CYAN).pack(anchor="w")
    tk.Label(panel_new, text="Enter driver name (max 30 chars):",
             font=FONT_SMALL, bg=C_PANEL, fg=C_GRAY).pack(anchor="w", pady=(6, 2))

    name_var = tk.StringVar()
    name_entry = tk.Entry(
        panel_new, textvariable=name_var,
        font=("Segoe UI", 12),
        bg=C_INPUT, fg=C_WHITE, insertbackground=C_CYAN,
        bd=0, highlightthickness=1,
        highlightcolor=C_CYAN, highlightbackground=C_PANEL,
        width=28,
    )
    name_entry.pack(anchor="w", ipady=5)

    def _limit_name(*_):
        v = name_var.get()
        if len(v) > 30:
            name_var.set(v[:30])
    name_var.trace_add("write", _limit_name)

    tk.Label(panel_new, text="Select age group:",
             font=FONT_SMALL, bg=C_PANEL, fg=C_GRAY).pack(anchor="w", pady=(14, 4))

    age_var = tk.StringVar(value="")
    age_btn_frame = tk.Frame(panel_new, bg=C_PANEL)
    age_btn_frame.pack(anchor="w")
    age_buttons: list[tk.Button] = []

    def select_age(age: str):
        age_var.set(age)
        for b in age_buttons:
            is_sel = b.cget("text") == age
            b.config(
                bg=C_CYAN if is_sel else C_INPUT,
                fg="#000000" if is_sel else C_WHITE,
            )

    for age in AGE_GROUPS:
        btn = tk.Button(
            age_btn_frame, text=age,
            font=FONT_BTN, bg=C_INPUT, fg=C_WHITE, bd=0,
            activebackground=C_CYAN, activeforeground="#000000",
            padx=10, pady=5,
            command=lambda a=age: select_age(a),
        )
        btn.pack(side=tk.LEFT, padx=(0, 6))
        age_buttons.append(btn)

    def cmd_create():
        global NO_DB_MODE
        name = name_var.get().strip()
        age  = age_var.get()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a driver name.", parent=root)
            return
        if not age:
            messagebox.showwarning("Missing Age Group", "Please select an age group.", parent=root)
            return

        driver_id = create_driver(name, age)

        if driver_id is None:
            # DB failed — activate No-DB mode instead of crashing
            NO_DB_MODE = True
            driver_id  = -1
            _log.error(
                "DB failed to create driver '%s'. Entering No-DB mode (session data will NOT be saved).",
                name,
            )
            messagebox.showwarning(
                "Database Error",
                "Could not save driver to database.\n\n"
                "The system will run in No-DB Mode — monitoring is active "
                "but session data will NOT be saved.",
                parent=root,
            )

        result[0] = (driver_id, name, age)
        root.destroy()

    tk.Button(
        panel_new, text="✚  Create & Start",
        font=FONT_BTN, bg=C_CYAN, fg="#000000", bd=0, relief=tk.FLAT,
        activebackground="#00aacc", activeforeground="#000000",
        padx=12, pady=6,
        command=cmd_create,
    ).pack(pady=(16, 0), anchor="e")

    # ── Show first panel ──────────────────────────────────────────────────────
    show_panel("existing" if all_drivers else "new")

    root.mainloop()

    if result[0] is None:
        sys.exit(0)

    return result[0]
