import cv2
import numpy as np
import sys
import json
import os

PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except: pass
    return {"Threshold": 194, "Min_Intensity": 139}

def save_config(thresh, min_int, roi_data):
    config = {"Threshold": thresh, "Min_Intensity": min_int}
    config.update(roi_data)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n[УСПЕХ] Пороги и ROI СЫРЫХ кадров сохранены в {CONFIG_PATH}!")
    print(f"LEFT ROI: [{roi_data['roi_l_start']}:{roi_data['roi_l_end']}]")
    print(f"RIGHT ROI: [{roi_data['roi_r_start']}:{roi_data['roi_r_end']}]")

def nothing(x):
    pass

saved_config = load_config()

cv2.namedWindow("TUNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("TUNER", 50, 550)
cv2.resizeWindow("TUNER", 400, 200)

cv2.createTrackbar("Threshold", "TUNER", saved_config.get("Threshold", 194), 255, nothing)
cv2.createTrackbar("Min_Intensity", "TUNER", saved_config.get("Min_Intensity", 139), 255, nothing)

cv2.namedWindow("Webcam LEFT (1)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam LEFT (1)", 50, 50)
cv2.namedWindow("Webcam RIGHT (0)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam RIGHT (0)", 750, 50)

cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)

cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    cv2.imshow("PROJECTOR", pattern)
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    
    if ret1 and ret2:
        thresh_val = cv2.getTrackbarPos("Threshold", "TUNER")
        min_int = cv2.getTrackbarPos("Min_Intensity", "TUNER")
        
        # РАБОТАЕМ С СЫРЫМИ КАДРАМИ! Никаких remap! Линия гарантированно на экране!
        b_channel1 = f1[:, :, 0]
        _, mask1 = cv2.threshold(b_channel1, thresh_val, 255, cv2.THRESH_BINARY)
        contour_f1 = f1.copy()
        contour_f1[mask1 > 0] = (0, 255, 0)
        
        b_channel2 = f2[:, :, 0]
        _, mask2 = cv2.threshold(b_channel2, thresh_val, 255, cv2.THRESH_BINARY)
        contour_f2 = f2.copy()
        contour_f2[mask2 > 0] = (0, 255, 0)

        # Твоя логика MIN/MAX по оси X (индекс 1)
        pts_l = np.where(mask1 > 0)
        pts_r = np.where(mask2 > 0)
        
        h, w = mask1.shape
        
        if len(pts_l) > 50:
            l_start = max(0, int(np.min(pts_l)) - 30) # Накинем 30px зазора под изгиб горба
            l_end = min(w, int(np.max(pts_l)) + 30)
        else:
            l_start, l_end = w // 2 - 50, w // 2 + 50

        if len(pts_r) > 50:
            r_start = max(0, int(np.min(pts_r)) - 30)
            r_end = min(w, int(np.max(pts_r)) + 30)
        else:
            r_start, r_end = w // 2 - 50, w // 2 + 50
        
        cv2.line(contour_f1, (l_start, 0), (l_start, h), (0, 255, 255), 2)
        cv2.line(contour_f1, (l_end, 0), (l_end, h), (0, 255, 255), 2)
        cv2.line(contour_f2, (r_start, 0), (r_start, h), (0, 255, 255), 2)
        cv2.line(contour_f2, (r_end, 0), (r_end, h), (0, 255, 255), 2)

        cv2.imshow("Webcam LEFT (1)", cv2.resize(contour_f1, (640, 360)))
        cv2.imshow("Webcam RIGHT (0)", cv2.resize(contour_f2, (640, 360)))
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_data = {
            "roi_l_start": l_start, "roi_l_end": l_end,
            "roi_r_start": r_start, "roi_r_end": r_end
        }
        save_config(thresh_val, min_int, roi_data)
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
