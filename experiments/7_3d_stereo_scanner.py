import cv2
import numpy as np
import json
import os
import sys

# --- НАСТРОЙКИ (можно вынести в JSON) ---
PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"
TOLERANCE_MM = 2.0
SMOOTH_FACTOR = 0.15 # Сглаживание 0..1
PLOT_SIZE = (500, 250) # (W, H)
REAL_BLOCK_MM = 15.0 # Для кнопки калибровки 'c'

print("=== ЗАПУСК ОБНОВЛЕННОГО 3D СТЕРЕО-СКАНЕРА С ОНЛАЙН ГРАФИКОМ ===")

# Загрузка настроек
if not os.path.exists(CONFIG_PATH):
    print(f"[ОШИБКА] Конфиг {CONFIG_PATH} не найден!")
    sys.exit(1)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Инициализация окон
# 2. Настройка окон интерфейса оператора пресса (Экран 3440х1440)
cv2.namedWindow("3D SCANNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("3D SCANNER", 50, 50)
cv2.resizeWindow("3D SCANNER", 1300, 480)

# Настройка проектора (Улетает на правый экран X=3440)
cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Маска с синей линией для проектора
pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)




# Камеры (индексы 1 и 0)
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def get_line_x(frame, cfg, side='l'):
    """Субпиксельный поиск линии в ROI"""
    r_start = cfg[f"roi_{side}_start"]
    r_end = cfg[f"roi_{side}_end"]
    b_channel = frame[:, :, 0]
    roi = b_channel[:, r_start:r_end]
    
    _, th = cv2.threshold(roi, cfg["Threshold"], 255, cv2.THRESH_TOZERO)
    th = th.astype(np.float32)
    
    sum_int = np.sum(th, axis=1)
    x_indices = np.arange(roi.shape[1], dtype=np.float32)
    sum_x_int = np.sum(th * x_indices, axis=1)
    
    centers = np.full(th.shape[0], -1.0, dtype=np.float32)
    mask = (sum_int > cfg["Min_Intensity"])
    centers[mask] = (sum_x_int[mask] / sum_int[mask]) + r_start
    return centers

# Переменные логики
base_disp = None
scale_mm_pix = 1.5
smoothed_dev = 0.0
history = []
status = "Нажми 'b' для калибровки нуля (стены)"

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    line_l = get_line_x(f1, config, 'l')
    line_r = get_line_x(f2, config, 'r')
    
    # Визуализация и расчет
    vis = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
    h, w = f1.shape[:2]
    current_disp = line_l - line_r
    raw_dev = 0.0
    plot_pts = []

    # Расчет дефектов
    for y in range(h):
        if line_l[y] > 0 and line_r[y] > 0:
            if base_disp is not None:
                delta = current_disp[y] - base_disp[y]
                z_mm = delta * scale_mm_pix
                if abs(z_mm) > abs(raw_dev): raw_dev = z_mm
                
                # Точки графика
                px = int((y / h) * PLOT_SIZE[0])
                py = int((PLOT_SIZE[1] // 2) - (z_mm * 4.0))
                plot_pts.append((px, max(10, min(PLOT_SIZE[1]-10, py))))

    # Отрисовка графика (аналог live_plots.py)
    plot_img = np.zeros((PLOT_SIZE[1], PLOT_SIZE[0], 3), dtype=np.uint8)
    cv2.line(plot_img, (0, PLOT_SIZE[1]//2), (PLOT_SIZE[0], PLOT_SIZE[1]//2), (100,100,100), 1)
    if len(plot_pts) > 1:
        cv2.polylines(plot_img, [np.array(plot_pts)], False, (0,255,0), 2)
    
    # История пиков
    if base_disp is not None and abs(raw_dev) > 0.1:
        history.append(raw_dev)
        if len(history) > PLOT_SIZE[0]: history.pop(0)
    for i, p in enumerate(history):
        py = int((PLOT_SIZE[1] // 2) - (p * 4.0))
        cv2.circle(plot_img, (i, max(10, min(PLOT_SIZE[1]-10, py))), 1, (0,165,255), -1)

    # Сглаживание и вывод
    smoothed_dev = (SMOOTH_FACTOR * raw_dev) + ((1.0 - SMOOTH_FACTOR) * smoothed_dev)
    
    # UI
    cv2.rectangle(vis, (0, h-PLOT_SIZE[1]), (PLOT_SIZE[0], h), (20,20,20), -1)
    vis[h-PLOT_SIZE[1]:h, 0:PLOT_SIZE[0]] = plot_img
    
    color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
    cv2.putText(vis, f"Dev: {smoothed_dev:.2f}mm | Scale: {scale_mm_pix:.3f}", (10, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("3D SCANNER", vis)
    cv2.imshow("PROJECTOR", pattern)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('b'):
        base_disp = current_disp.copy()
        status = "База установлена (0 мм)"
    elif key == ord('c'):
        if base_disp is not None and abs(smoothed_dev) > 0.1:
            scale_mm_pix = (REAL_BLOCK_MM / abs(smoothed_dev)) * scale_mm_pix
            status = f"Калибровка: {scale_mm_pix:.4f} мм/пикс"
        else: status = "Ошибка калибровки (нужен брусок)"

cap1.release()
cap2.release()
cv2.destroyAllWindows()
