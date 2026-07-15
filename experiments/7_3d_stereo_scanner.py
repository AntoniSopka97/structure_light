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
REAL_BLOCK_MM = 15.0 # Для кнопки калибровки 'c'

# Новые размеры для большого графика
WINDOW_W = 1080
WINDOW_H = 600
GRAPH_H = 400  # Высота графика (было 250)
VIDEO_H = 340   # Высота видео
STATUS_H = 50   # Высота статуса
PANEL_W = WINDOW_W  # Ширина всех элементов

print("=== ЗАПУСК ОБНОВЛЕННОГО 3D СТЕРЕО-СКАНЕРА С ОНЛАЙН ГРАФИКОМ ===")

# Загрузка настроек
if not os.path.exists(CONFIG_PATH):
    print(f"[ОШИБКА] Конфиг {CONFIG_PATH} не найден!")
    sys.exit(1)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Инициализация окон (НОВЫЙ РАЗМЕР)
cv2.namedWindow("3D SCANNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("3D SCANNER", 50, 50)
cv2.resizeWindow("3D SCANNER", WINDOW_W, WINDOW_H)

# Настройка проектора
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

    # Расчет дефектов (ЛОГИКА НЕ ТРОГАЕТСЯ)
    for y in range(h):
        if line_l[y] > 0 and line_r[y] > 0:
            if base_disp is not None:
                delta = current_disp[y] - base_disp[y]
                z_mm = delta * scale_mm_pix
                if abs(z_mm) > abs(raw_dev): raw_dev = z_mm
                
                # Точки графика - теперь на всю ширину
                px = int((y / h) * PANEL_W)
                py = int((GRAPH_H // 2) - (z_mm * 4.0))
                plot_pts.append((px, max(10, min(GRAPH_H-10, py))))

    # --- БОЛЬШОЙ КРАСИВЫЙ ГРАФИК НА ВСЮ ШИРИНУ ---
    plot_img = np.zeros((GRAPH_H, PANEL_W, 3), dtype=np.uint8)
    plot_img.fill(25)  # Темный фон
    
    # Сетка
    for y in range(0, GRAPH_H, 40):
        cv2.line(plot_img, (0, y), (PANEL_W, y), (45, 45, 45), 1)
    for x in range(0, PANEL_W, 60):
        cv2.line(plot_img, (x, 0), (x, GRAPH_H), (45, 45, 45), 1)
    
    # Нулевая линия (жирная)
    cv2.line(plot_img, (0, GRAPH_H//2), (PANEL_W, GRAPH_H//2), (100, 100, 100), 2)
    
    # Зона допуска (полупрозрачная полоса)
    tol_pixels = int(TOLERANCE_MM * 4.0)
    overlay = plot_img.copy()
    cv2.rectangle(overlay, (0, GRAPH_H//2 - tol_pixels), (PANEL_W, GRAPH_H//2 + tol_pixels), (0, 80, 0), -1)
    cv2.addWeighted(overlay, 0.2, plot_img, 0.8, 0, plot_img)
    
    # Основной профиль (зеленый/красный)
    if len(plot_pts) > 1:
        color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
        cv2.polylines(plot_img, [np.array(plot_pts)], False, color, 3, cv2.LINE_AA)
        
        # Тень под линией
        fill_pts = np.array(plot_pts + [(PANEL_W-1, GRAPH_H//2), (0, GRAPH_H//2)])
        cv2.fillPoly(plot_img, [fill_pts], (color[0]//4, color[1]//4, color[2]//4))
    
    # История пиков (оранжевые точки)
    for i, p in enumerate(history):
        py = int((GRAPH_H // 2) - (p * 4.0))
        py = max(10, min(GRAPH_H-10, py))
        if i < PANEL_W:
            cv2.circle(plot_img, (i, py), 2, (0, 165, 255), -1)
    
    # Подписи значений на осях
    cv2.putText(plot_img, f"+{TOLERANCE_MM:.1f}mm", (10, GRAPH_H//2 - tol_pixels - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(plot_img, f"-{TOLERANCE_MM:.1f}mm", (10, GRAPH_H//2 + tol_pixels + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    # Текущее значение (крупно в углу)
    val_color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
    cv2.putText(plot_img, f"{smoothed_dev:+.2f} mm", (PANEL_W-180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, val_color, 3)
    
    # Масштаб и точки
    cv2.putText(plot_img, f"Scale: {scale_mm_pix:.3f} mm/px", (PANEL_W-180, GRAPH_H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(plot_img, f"Points: {len(plot_pts)}", (10, GRAPH_H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # --- ИСТОРИЯ ПИКОВ (ЛОГИКА НЕ ТРОГАЕТСЯ) ---
    if base_disp is not None and abs(raw_dev) > 0.1:
        history.append(raw_dev)
        if len(history) > PANEL_W: history.pop(0)

    # --- СГЛАЖИВАНИЕ (ЛОГИКА НЕ ТРОГАЕТСЯ) ---
    smoothed_dev = (SMOOTH_FACTOR * raw_dev) + ((1.0 - SMOOTH_FACTOR) * smoothed_dev)
    
    # --- ВИДЕО ПАНЕЛЬ (ТЕПЕРЬ ТОЖЕ БОЛЬШАЯ) ---
    vis_l = cv2.resize(f1, (PANEL_W//2, VIDEO_H))
    vis_r = cv2.resize(f2, (PANEL_W//2, VIDEO_H))
    video_panel = np.hstack((vis_l, vis_r))
    
    # --- СТАТУСНАЯ СТРОКА ---
    status_bar = np.zeros((STATUS_H, PANEL_W, 3), dtype=np.uint8)
    status_bar.fill(20)
    cv2.putText(status_bar, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Индикатор статуса
    color = (0, 255, 0) if base_disp is not None else (0, 0, 255)
    cv2.circle(status_bar, (PANEL_W-40, 25), 10, color, -1)
    
    # --- СБОРКА ВСЕГО ОКНА ---
    top = np.vstack((video_panel, status_bar))
    full = np.vstack((top, plot_img))
    
    # Рамка
    cv2.rectangle(full, (0, 0), (full.shape[1]-1, full.shape[0]-1), (60, 60, 60), 2)
    
    cv2.imshow("3D SCANNER", full)
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