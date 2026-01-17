import cv2
import numpy as np

def extract_features(video):
    cap = cv2.VideoCapture(video)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motions, densities = [], []
    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motions.append(np.mean(mag))

        fg = backSub.apply(frame)
        density = np.sum(fg > 0) / fg.size
        densities.append(density)

        prev_gray = gray

    cap.release()
    return np.mean(motions), np.mean(densities)

normal_motion, normal_density = extract_features("videos/normal.mp4")
abnormal_motion, abnormal_density = extract_features("videos/abnormal.mp4")

LOW_MOTION = normal_motion
MEDIUM_MOTION = (normal_motion + abnormal_motion) / 2
LOW_DENSITY = normal_density
HIGH_DENSITY = abnormal_density

print("=== TRAINED THRESHOLDS ===")
print("LOW_MOTION:", LOW_MOTION)
print("MEDIUM_MOTION:", MEDIUM_MOTION)
print("LOW_DENSITY:", LOW_DENSITY)
print("HIGH_DENSITY:", HIGH_DENSITY)
