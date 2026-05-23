# WakeMate v2.0 — Real-Time Driver Fatigue Detection System

> **⚠ Safety Notice:** This system is a research and assistive tool.  
> It does not replace human judgment or safe driving practices.

WakeMate monitors a driver's face in real time using a webcam, detecting drowsiness, yawning, head-drop, and distraction. When fatigue is detected it sounds an audio alarm, records a short video clip, and logs the event to an SQLite database.

---

## What's New in v2.0

| Feature | Detail |
|---------|--------|
| **DetectionEngine** | All camera/detection logic extracted into one class — no more 200-line code duplication |
| **SQLite-Primary** | SQLite is now the single source of truth; JSON is exported on demand |
| **Export (JSON / CSV)** | One-click export from the History tab for offline analysis |
| **Clip Playback** | "▶ Play Clip" button opens recorded drowsiness clips in your system media player |
| **No-DB Mode** | DB failures show a visible amber banner instead of crashing |
| **Break Reminder** | One-time popup after 120 minutes of continuous driving |
| **Audio Fallback** | Missing WAV files fall back to a system beep — no crash |
| **Clip Cleanup** | Oldest clips are auto-deleted on startup to cap disk usage at 50 clips |
| **App Logging** | All events written to `wakemate.log` with timestamps and log levels |
| **Version Branding** | Window title shows `WAKEMATE v2.0.0` |

---

## Architecture

```
dashboard.py          main.py
     │                    │
     │   (Tkinter GUI)    │  (OpenCV/CLI)
     └────────┬───────────┘
              │ instantiates
              ▼
    detection_engine.py   ◄── detector.py  (EAR / MAR / head pose)
         │      │              thresholds.py (adaptive EAR/MAR)
         │      │              alarms.py      (audio + fallback)
         │      │              clip_recorder.py (MP4 clips)
         │      └──────────►  logger.py      (in-memory + DB)
         │
         ▼
      database.py (SQLite: drivers / sessions / events)
         │
         ▼
      exporter.py  (JSON / CSV on demand)
      analytics.py (risk scoring, trends)
      app_logger.py (rotating log file → wakemate.log)
```

---

## Requirements

### Python
Python **3.10+** is required (uses `match` expressions and `X | Y` union types).

### Windows — Installing cmake and dlib

`dlib` requires a C++ compiler and `cmake`.

1. **Install Visual Studio Build Tools** (C++ workload):  
   Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. **Install cmake**:  
   ```powershell
   winget install Kitware.CMake
   ```
   Or download the installer from https://cmake.org/download/

3. Restart your terminal so `cmake` is on `PATH`.

---

## Installation

```powershell
# 1. Clone / unzip the project
cd Real-Time---Driver-Fatigue-Detection-System-master

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the Dlib landmark model (~99 MB)
python download_model.py
```

### Required Audio Assets

The alarm sounds are included in the repository:

| File | Used for |
|------|----------|
| `sound.wav` | Eye closure / head-down / distraction alarms |
| `music.wav` | Yawn alarm |

If either file is **missing**, WakeMate falls back to a system beep and logs a warning to `wakemate.log` — it will not crash.

---

## Usage

### Dashboard (recommended)
```powershell
python dashboard.py
```

### CLI / OpenCV window
```powershell
python main.py
```

### First Launch

1. The **Driver Setup** dialog appears.
2. Select an existing driver or create a new profile (name + age group).
3. The camera opens and a **5-minute calibration phase** begins.  
   Sit normally with eyes open so WakeMate learns your baseline EAR/MAR.
4. After calibration, real-time monitoring starts automatically.

---

## Dashboard Guide

### 📷 Live Monitor Tab

| Widget | Description |
|--------|-------------|
| **Live Feed** | Camera stream with landmark overlay. Flashes red on alert. |
| **Driver Stats** | EAR, MAR, yawn count, drive time, status |
| **Alerts (Last 5)** | Most recent fatigue events with timestamps |

### 📊 History Tab

Shows all past sessions for the selected driver with risk scores.

| Button | Action |
|--------|--------|
| **⬇ Export JSON** | Save full session data (metadata + events) as `.json` |
| **⬇ Export CSV** | Save event log as flat `.csv` for spreadsheet analysis |
| **▶ Play Clip** | Open the most recent drowsiness clip in your system media player |

> Select a row first to enable the buttons.

### Control Panel

| Button | Action |
|--------|--------|
| **▶ Start** | Resume after pause |
| **⏸ Pause** | Pause detection (clock stops, no alarms) |
| **⏹ Stop** | End session, save to DB, refresh history |

---

## Detection Logic

### Eye Aspect Ratio (EAR)
```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
```
- Calibrated per-driver during the 5-minute startup phase
- **Alarm** fires if EAR < threshold for **≥1.5 seconds**

### Mouth Aspect Ratio (MAR)
```
MAR = (Y1 + Y2) / (2 · X)    where X = mouth width
```
- **Yawn alarm** fires after **3 yawns ≥ 1.5 s** each

### Head Pose
Solved via `cv2.solvePnP` with 14 facial landmarks:
- **Head down** → pitch angle X > 10°
- **Distracted** → yaw angle |Y| > 20°

### Adaptive Thresholds
Every 30 minutes, EAR/MAR thresholds are recalculated based on:
- Age group factor (older → more sensitive)
- Drive duration factor (longer drive → more sensitive)

### Risk Score (0–100)

| Event | Points | Cap |
|-------|--------|-----|
| Drowsiness alarm | 25 | 100 |
| Yawn detected | 10 | 40 |
| Head down | 5 | 30 |
| Distracted | 5 | 30 |

---

## Data & Files

| File | Purpose |
|------|---------|
| `fatigue.db` | SQLite database (drivers, sessions, events) |
| `wakemate.log` | Application log (rotating, max 5 MB × 3 backups) |
| `clips/` | Short MP4 clips triggered by drowsiness alarms |

### Database Schema

```sql
drivers  (id, name, age_group, created_at)
sessions (id, driver_id, start_time, end_time, total_drive_mins,
          baseline_ear, baseline_mar, risk_score, age_group)
events   (id, session_id, event_type, timestamp, drive_minute,
          metric_value, threshold_value, clip_path)
```

---

## Troubleshooting

### Camera won't open
WakeMate tries 7 camera combinations (DirectShow, MSMF, Auto × indices 0–2).
Check `wakemate.log` for which ones were attempted and why they failed.

### `dlib` build fails
Ensure cmake is installed and on PATH:
```powershell
cmake --version   # should print version
```
Then reinstall dlib:
```powershell
pip install dlib --no-cache-dir
```

### No alarm sound
Check that `sound.wav` and `music.wav` exist in the project root.  
A system beep fallback is used automatically if they are missing.

### History tab is empty
The History tab shows sessions for the **currently selected driver**.  
Sessions from a different driver profile will not appear.

---

## Project Structure (v2.0)

```
├── dashboard.py          # Tkinter GUI entry point
├── main.py               # OpenCV/CLI entry point  
├── detection_engine.py   # 🆕 Core detection class (replaces duplication)
├── app_logger.py         # 🆕 Centralised logging → wakemate.log
├── exporter.py           # 🆕 On-demand JSON/CSV export
├── download_model.py     # 🆕 One-command model downloader
├── detector.py           # EAR / MAR / head pose math
├── thresholds.py         # Adaptive threshold functions
├── alarms.py             # Audio alarms with fallback
├── clip_recorder.py      # MP4 clip recording + cleanup
├── logger.py             # In-memory session logger + DB bridge
├── database.py           # SQLite CRUD layer
├── driver_manager.py     # Driver setup dialog
├── analytics.py          # Risk scoring and trend analysis
├── requirements.txt      # Python dependencies
├── fatigue.db            # SQLite database (auto-created)
├── wakemate.log          # Application log (auto-created)
├── clips/                # Drowsiness video clips
├── sound.wav             # Eye/head/distraction alarm
└── music.wav             # Yawn alarm
```

---

## License

This project is released for educational and research purposes.  
See the original repository for licensing details.
