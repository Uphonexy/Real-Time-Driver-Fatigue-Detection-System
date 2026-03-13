def compute_adaptive_threshold(personal_baseline, age_group, drive_minutes):
    """Compute adaptive threshold based on age factor and drive duration."""
    # Age factor
    if age_group == "18-30": age_factor = 1.0
    elif age_group == "31-45": age_factor = 1.08
    elif age_group == "46-60": age_factor = 1.18
    else: age_factor = 1.30

    # Time factor
    if drive_minutes < 120: time_factor = 1.0
    elif drive_minutes < 240: time_factor = 1.12
    elif drive_minutes < 360: time_factor = 1.25
    else: time_factor = 1.40

    # Threshold Multiplier (less scaling for EAR since it drops when eyes close)
    # The higher the age, the lower the natural EAR droop, so we RAISE the threshold.
    # The longer the drive, the lower the threshold so it catches smaller blinks, but we want it higher to be MORE sensitive.
    # Therefore, older age -> higher threshold. Longer drive -> higher threshold.
    return personal_baseline * age_factor * time_factor

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
