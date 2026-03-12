import time
import cv2
import imutils

def run_calibration_phase(cap):
    """Wait for user to select an age group, then return the selection and start time."""
    print("[INFO] Waiting for age group selection...")
    age_group = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame.")
            exit()

        frame = imutils.resize(frame, width=640)

        cv2.putText(frame, "Select Age Group:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 1: 18-30", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 2: 31-45", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 3: 46-60", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 4: 60+", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pre-Detection Setup", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            age_group = "18-30"
            break
        elif key == ord('2'):
            age_group = "31-45"
            break
        elif key == ord('3'):
            age_group = "46-60"
            break
        elif key == ord('4'):
            age_group = "60+"
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Pre-Detection Setup")
    print(f"[INFO] Selected age group: {age_group}")
    
    return age_group, time.time()
