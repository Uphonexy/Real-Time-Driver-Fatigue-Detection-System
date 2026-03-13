import json
from datetime import datetime

class SessionLogger:
    def __init__(self, age_group):
        self.session_log = []
        self.session_meta = {
            "age_group": age_group,
            "drive_start_time": datetime.now().isoformat(),
            "personal_baseline_ear": None,
            "personal_baseline_mar": None # FIX 5: MAR calibration from samples
        }

    def update_baseline(self, baseline):
        self.session_meta["personal_baseline_ear"] = baseline

    # FIX 5: Update MAR baseline in logger
    def update_baseline_mar(self, baseline_mar):
        self.session_meta["personal_baseline_mar"] = baseline_mar

    def log_event(self, event_type, drive_minutes, metric_value, threshold_at_event):
        self.session_log.append({
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "drive_minutes": round(drive_minutes, 2),
            "metric_value": round(float(metric_value), 3),
            "threshold_at_event": round(float(threshold_at_event), 3)
        })

    def save_and_print_summary(self, total_drive_time):
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_log = {
            "session_meta": self.session_meta,
            "events": self.session_log
        }

        with open(filename, 'w') as f:
            json.dump(full_log, f, indent=4)

        eyes_closed_count = sum(1 for e in self.session_log if e['event_type'] == 'eyes_closed')
        drowsy_alarms = sum(1 for e in self.session_log if e['event_type'] == 'drowsiness_alarm')
        yawn_alerts = sum(1 for e in self.session_log if e['event_type'] == 'yawn_detected')
        head_downs = sum(1 for e in self.session_log if e['event_type'] == 'head_down')
        distracted = sum(1 for e in self.session_log if e['event_type'] == 'distracted')

        print("\n" + "="*40)
        print("     DRIVER SESSION SUMMARY")
        print("="*40)
        print(f"Age Group:             {self.session_meta['age_group']}")
        print(f"Total Drive Time:      {total_drive_time} mins")
        print(f"Baseline EAR:          {self.session_meta['personal_baseline_ear']}")
        print(f"Baseline MAR:          {self.session_meta['personal_baseline_mar']}") # FIX 5: Display MAR baseline
        print(f"\n--- Events ---")
        print(f"Eyes Closed Events:    {eyes_closed_count}")
        print(f"Head Down Alerts:      {head_downs}")
        print(f"Distracted Alerts:     {distracted}")
        print(f"Yawn Cycle Alerts:     {yawn_alerts}")
        print(f"Total Drowsy Alarms:   {drowsy_alarms}")
        print("="*40)
        print(f"[INFO] Log saved to {filename}")
