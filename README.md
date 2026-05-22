# Real-Time Driver Fatigue Detection System — WAKEMATE

A Python-based real-time drowsiness and distraction detection system that uses facial landmarks to monitor eye closure, yawning, and head pose. The system triggers audio alarms, records short video clips at alarm events, logs all session data to a SQLite database, and provides a rich Tkinter dashboard with driver profiles and session history.

---

## Features

### Core Detection
- **Eye Closure Detection** — Real-time Eye Aspect Ratio (EAR) monitoring; alarm fires after eyes are closed ≥ 1.5 seconds
- **Yawning Detection** — Mouth Aspect Ratio (MAR) monitoring; three confirmed yawn cycles trigger an alert
- **Head Pose & Distraction Monitoring** — Detects head pitch (nodding) and yaw (looking away) via 3D pose estimation
- **Personalized Calibration** — 5-minute initialization phase builds a unique EAR/MAR baseline per driver
- **Adaptive Thresholds** — Sensitivity adjusts automatically based on age group and cumulative drive time
- **Distinct Audio Alerts** — Separate alarm sounds for drowsiness, yawning, head-down, and distraction events

### New in This Release
- **Driver Profiles** — Named driver accounts stored in SQLite; returning drivers are recognized across sessions
- **Session History** — Every drive is recorded (start/end time, drive duration, baselines, risk score) and viewable in the dashboard History tab
- **Clip Recording** — A 4-second MP4 clip (1 s pre-alarm + 3 s post-alarm) is saved automatically every time a drowsiness alarm fires, at 480×360 / 10 fps (~300–600 KB per clip)
- **Risk Scoring** — Each session receives a 0–100 risk score (Low / Moderate / High / Critical) computed from weighted event counts
- **📊 History Tab** — Treeview in the dashboard listing all past sessions with color-coded risk labels
- **Pause-Aware Drive Time** — Drive timer correctly excludes paused intervals in both the dashboard and the OpenCV window

---

## Project Structure

```
project/
├── main.py              Entry point — OpenCV detection window
├── dashboard.py         Tkinter GUI dashboard with live feed & history tab
│
├── database.py          SQLite persistence layer (fatigue.db)
├── analytics.py         Risk scoring, session summaries, driver trend data
├── clip_recorder.py     Drowsiness alarm clip recorder (OpenCV VideoWriter)
├── driver_manager.py    Driver setup Tkinter dialog (new / existing driver)
│
├── detector.py          EAR, MAR, head-pose math
├── thresholds.py        Adaptive EAR/MAR threshold computation
├── calibration.py       Age-group selection for main.py (OpenCV key-press UI)
│                        ⚠ Superseded in dashboard.py by driver_manager.py
├── alarms.py            Pygame audio alerts with cooldown
├── logger.py            JSON session logging + DB event sink
│
├── requirements.txt
├── fatigue.db           ← auto-created at runtime (git-ignored)
└── clips/               ← auto-created at runtime (git-ignored)
    └── clip_YYYYMMDD_HHMMSS.mp4
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture, frame processing, clip writing |
| `dlib` | Facial landmark detection |
| `numpy` | Numerical arrays |
| `imutils` | Frame resizing helpers |
| `pygame` | Audio alarm playback |
| `Pillow` | Tkinter image display in dashboard |
| `scipy` | Geometry calculations in detector.py |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Uphonexy/Real-Time-Driver-Fatigue-Detection-System.git
   cd Real-Time-Driver-Fatigue-Detection-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the facial landmark model**
   Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project root.

4. **Run the application**

   **Dashboard mode** (recommended — full GUI with history and driver profiles):
   ```bash
   python dashboard.py
   ```

   **OpenCV mode** (lightweight, keyboard-only):
   ```bash
   python main.py
   ```

---

## How It Works

### Startup flow
1. The **Driver Setup** dialog appears — create a new driver (name + age group) or select an existing one.
2. A new session row is created in `fatigue.db`.
3. The camera opens and a **5-minute calibration phase** begins — sit normally and keep your eyes open.
4. After calibration, adaptive thresholds are set and monitoring begins.

### During a drive
| Event | Action |
|---|---|
| Eyes closed ≥ 1.5 s | 🔊 Alarm + 4-second MP4 clip saved to `clips/` |
| 3 confirmed yawns | 🔊 Alarm + event logged |
| Head pitched down | 🔊 Alarm + event logged |
| Looking away | 🔊 Alarm + event logged |

### Session end
- Press **⏹ Stop** (dashboard) or **Q** (OpenCV window)
- A `session_YYYYMMDD_HHMMSS.json` file is saved (full event log)
- The session row in `fatigue.db` is closed with end time, drive duration, and risk score
- The **📊 History** tab updates with the new session

### Risk Score
```
drowsiness_alarm  ×25 pts  (cap 100)
yawn_detected     ×10 pts  (cap  40)
head_down          ×5 pts  (cap  30)
distracted         ×5 pts  (cap  30)
─────────────────────────────────────
Final = min(100, sum)

0–25   → LOW RISK      (green)
26–50  → MODERATE RISK (amber)
51–75  → HIGH RISK     (red)
76–100 → CRITICAL RISK (red, bold)
```

---

## Controls

### Dashboard
| Button | Action |
|---|---|
| ▶ Start | Resume detection after pause |
| ⏸ Pause | Pause detection (timer freezes) |
| ⏹ Stop | End session, save logs |
| 🔄 Refresh History | Reload session history from DB |

### OpenCV window (`main.py`)
| Key | Action |
|---|---|
| `P` | Pause / Resume |
| `Q` | Quit and save session |

---

## License

This project is licensed under the MIT License.
