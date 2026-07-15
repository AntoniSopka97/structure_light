import cv2
import numpy as np
import json
import os
import sys
from collections import deque
import time

# --- НАСТРОЙКИ ---
PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"
TOLERANCE_MM = 2.0
SMOOTH_FACTOR = 0.3
PLOT_SIZE = (500, 250)
HISTORY_LEN = 150

print("=== ЗАПУСК СТЕРЕО-СКАНЕРА С ПРОДВИНУТОЙ ВИЗУАЛИЗАЦИЕЙ ===")

# --- ЗАГРУЗКА КОНФИГА ---
if not os.path.exists(CONFIG_PATH):
    print(f"[ОШИБКА] Конфиг {CONFIG_PATH} не найден!")
    sys.exit(1)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- ИНИЦИАЛИЗАЦИЯ ОКОН ---
# Главное окно (видео + график)
cv2.namedWindow("3D SCANNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("3D SCANNER", 50, 50)
cv2.resizeWindow("3D SCANNER", 1300, 780)

# Окно проектора
cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 8)

# --- ЗАПУСК КАМЕР ---
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_line_x(frame, cfg, side='l'):
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

def draw_dashboard(plot_img, raw_dev, smoothed_dev, scale, status, history, tolerance):
    """Отрисовка современной приборной панели."""
    h, w = plot_img.shape[:2]
    
    # 1. График профиля с цветовой индикацией
    cv2.rectangle(plot_img, (0, 0), (w, h), (40, 40, 45), -1)  # Темный фон
    # Оси и сетка
    for i in range(0, h, 30):
        cv2.line(plot_img, (0, i), (w, i), (60, 60, 70), 1)
    cv2.line(plot_img, (0, h//2), (w, h//2), (100, 100, 120), 2)
    
    # Линия профиля с градиентом цвета (зеленый -> красный)
    if len(history) > 1:
        pts = np.array([[(i, int(h//2 - val * 3.5)) for i, val in enumerate(history)]], dtype=np.int32)
        # Цвет линии зависит от последнего значения
        color = (0, 255, 0) if abs(smoothed_dev) < tolerance else (0, 0, 255)
        cv2.polylines(plot_img, pts, False, color, 2, cv2.LINE_AA)
        # Заполнение под графиком (прозрачный полигон)
        if len(pts[0]) > 0:
            fill_pts = np.concatenate([pts[0], [[w-1, h//2], [0, h//2]]])
            cv2.fillPoly(plot_img, [fill_pts], color=(*color, 0.3), lineType=cv2.LINE_AA)

    # 2. Цифровые индикаторы
    # Статус
    cv2.rectangle(plot_img, (10, 10), (350, 45), (20, 20, 25), -1)
    cv2.putText(plot_img, f"СТАТУС: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Отклонение с цветным фоном
    dev_color = (0, 255, 0) if abs(smoothed_dev) < tolerance else (0, 0, 255)
    cv2.rectangle(plot_img, (w-230, 10), (w-10, 45), (20, 20, 25), -1)
    cv2.rectangle(plot_img, (w-230, 10), (w-10, 45), dev_color, 2)
    cv2.putText(plot_img, f"{smoothed_dev:+.2f} мм", (w-200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dev_color, 2)
    
    # Масштаб и точки
    cv2.putText(plot_img, f"Масштаб: {scale:.3f} мм/пкс | Точек: {len(history)}", (w-480, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1)

    return plot_img

def apply_thermal_overlay(frame, z_data, alpha=0.3):
    """Наложение термокарты на изображение."""
    if z_data is None or len(z_data) == 0:
        return frame
    # Нормализуем z_data для цветовой карты (упрощенно)
    z_norm = cv2.normalize(z_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Применяем цветовую карту JET
    heatmap = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
    # Накладываем на кадр
    overlay = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
    return overlay

# --- ОСНОВНОЙ ЦИКЛ ---
base_disp = None
scale_mm_pix = 1.2
smoothed_dev = 0.0
history = deque(maxlen=HISTORY_LEN)
status = "Нажмите 'b' для калибровки нуля (фон)"
z_data_prev = None
frame_counter = 0

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: 
        continue

    frame_counter += 1
    line_l = get_line_x(f1, config, 'l')
    line_r = get_line_x(f2, config, 'r')
    
    # --- ОСНОВНАЯ ОБРАБОТКА ---
    h, w = f1.shape[:2]
    current_disp = line_l - line_r
    raw_dev = 0.0
    z_values = np.zeros(h, dtype=np.float32)

    # Вычисление профиля высот
    if base_disp is not None:
        for y in range(h):
            if line_l[y] > 0 and line_r[y] > 0:
                delta = current_disp[y] - base_disp[y]
                z_mm = delta * scale_mm_pix
                z_values[y] = z_mm
                if abs(z_mm) > abs(raw_dev): 
                    raw_dev = z_mm
    
    # Сглаживание
    smoothed_dev = (SMOOTH_FACTOR * raw_dev) + ((1.0 - SMOOTH_FACTOR) * smoothed_dev)
    if abs(smoothed_dev) > 0.05:
        history.append(smoothed_dev)
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    # 1. Основное изображение (с термокартой и аннотациями)
    display = cv2.resize(f1, (640, 360))
    display_r = cv2.resize(f2, (640, 360))
    vis = np.hstack((display, display_r))
    vis = cv2.resize(vis, (0,0), fx=1.2, fy=1.2)  # Увеличим для читаемости
    
    # Наложение тепловой карты (если есть данные)
    if base_disp is not None and np.max(np.abs(z_values)) > 0.1:
        # Для демонстрации используем массив z_values
        z_for_heatmap = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 0.001) * 255
        z_for_heatmap = z_for_heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(z_for_heatmap, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (vis.shape[1], vis.shape[0]))
        vis = cv2.addWeighted(vis, 0.6, heatmap_resized, 0.4, 0)
    
    # Отображение линии ROI и центров
    cv2.line(vis, (int(config["roi_l_start"]/3), 0), (int(config["roi_l_start"]/3), vis.shape[0]), (0, 255, 0), 1)
    cv2.line(vis, (int(config["roi_l_end"]/3), 0), (int(config["roi_l_end"]/3), vis.shape[0]), (0, 255, 0), 1)
    
    # 2. Приборная панель (в отдельном окне)
    plot_img = np.zeros((PLOT_SIZE[1], PLOT_SIZE[0], 3), dtype=np.uint8)
    plot_img = draw_dashboard(plot_img, raw_dev, smoothed_dev, scale_mm_pix, status, history, TOLERANCE_MM)
    
    # 3. Объединение и отображение
    # Добавляем график внизу главного окна
    plot_resized = cv2.resize(plot_img, (vis.shape[1], int(vis.shape[0]*0.35)))
    vis_final = np.vstack((vis, plot_resized))
    
    # Индикатор допуска в углу
    if base_disp is not None:
        color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
        cv2.circle(vis_final, (vis_final.shape[1]-30, 30), 15, color, -1)
        cv2.putText(vis_final, f"{abs(smoothed_dev):.1f} мм", (vis_final.shape[1]-80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    else:
        cv2.putText(vis_final, "НЕ ОТКАЛИБРОВАН", (vis_final.shape[1]-180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("3D SCANNER", vis_final)
    cv2.imshow("PROJECTOR", pattern)

    # --- УПРАВЛЕНИЕ ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('b'):
        if np.count_nonzero(current_disp > 0) > 100:
            base_disp = current_disp.copy()
            history.clear()
            status = "База установлена. Вносите деталь."
            print("[КАЛИБРОВКА] Ноль установлен.")
        else:
            status = "ОШИБКА: Не вижу линию!"
    elif key == ord('c'):
        # Пример калибровки масштаба по эталону (15 мм)
        if base_disp is not None and abs(smoothed_dev) > 0.5:
            scale_mm_pix = (15.0 / abs(smoothed_dev)) * scale_mm_pix
            status = f"Масштаб обновлен: {scale_mm_pix:.3f} мм/пикс"
            print(f"[КАЛИБРОВКА] Новый масштаб: {scale_mm_pix:.3f}")
        else:
            status = "Ошибка: поместите эталон 15 мм в луч"
    elif key == ord('r'):
        history.clear()
        status = "История сброшена"

# --- ЗАВЕРШЕНИЕ ---
cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("=== РАБОТА ЗАВЕРШЕНА ===")