import cv2
import numpy as np
import time
import config

cap = cv2.VideoCapture("videos/abnormal.mp4")

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

backSub = cv2.createBackgroundSubtractorMOG2()

print("AI Public Safety Monitoring Started...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    avg_motion = np.mean(mag)

    fg = backSub.apply(frame)
    density = np.sum(fg > 0) / fg.size

    if avg_motion > config.MEDIUM_MOTION or density > config.HIGH_DENSITY:
        risk = "HIGH"
        reason = "Sudden crowd acceleration or congestion"
    elif avg_motion > config.LOW_MOTION:
        risk = "MEDIUM"
        reason = "Increased crowd activity detected"
    else:
        risk = "LOW"
        reason = "Normal crowd movement"

    timestamp = time.strftime("%H:%M:%S")

    if risk != "LOW":
        print(f"[{timestamp}] RISK LEVEL: {risk}")
        print(f"Motion: {avg_motion:.2f}, Density: {density:.3f}")
        print(f"Reason: {reason}\n")

    prev_gray = gray

cap.release()
print("Monitoring Ended.")
