def compute_adaptive_threshold(personal_baseline, age_group, drive_minutes):
    """
    Compute adaptive EAR threshold.
    Older drivers have lower resting EAR (ptosis) — we LOWER the threshold
    (more sensitive) for older groups, consistent with Dinges et al.
    Longer drives also lower the threshold (more sensitive) as fatigue accumulates.
    """
    if age_group == "18-30":   age_factor = 1.00
    elif age_group == "31-45": age_factor = 1.05
    elif age_group == "46-60": age_factor = 1.12
    else:                      age_factor = 1.20

    if drive_minutes < 120:    time_factor = 1.00
    elif drive_minutes < 240:  time_factor = 1.08
    elif drive_minutes < 360:  time_factor = 1.15
    else:                      time_factor = 1.25

    # CORRECTED: divide so older/longer → lower threshold → more sensitive
    return personal_baseline / (age_factor * time_factor)

# FIX 4: Separate MAR threshold logic (divides by factors)
def compute_adaptive_mar_threshold(personal_baseline, age_group, drive_minutes):
    """Compute adaptive MAR threshold (divides by age/time factors to lower threshold)."""
    if age_group == "18-30": age_factor = 1.0
    elif age_group == "31-45": age_factor = 1.08
    elif age_group == "46-60": age_factor = 1.18
    else: age_factor = 1.30

    if drive_minutes < 120: time_factor = 1.0
    elif drive_minutes < 240: time_factor = 1.12
    elif drive_minutes < 360: time_factor = 1.25
    else: time_factor = 1.40

    return personal_baseline / (age_factor * time_factor)
