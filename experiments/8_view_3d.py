import cv2
import numpy as np
import json
import os
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- НАСТРОЙКИ ---
PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"
TOLERANCE_MM = 2.0
SMOOTH_FACTOR = 0.15
REAL_BLOCK_MM = 15.0

# Размеры окна
WINDOW_W = 1080
WINDOW_H = 600
GRAPH_H = 320
VIDEO_H = 230
STATUS_H = 50
PANEL_W = WINDOW_W

# Настройки записи
RECORD_SECONDS = 10.0
FPS = 30
MAX_FRAMES = int(RECORD_SECONDS * FPS)

print("=== 3D СТЕРЕО-СКАНЕР С ЗАПИСЬЮ 3D ===")
print("Управление:")
print("  'b' - калибровка нуля (стена)")
print("  'r' - запись 5 секунд (двигай деталь!)")
print("  'c' - калибровка масштаба")
print("  'ESC' - выход")

# Загрузка настроек
if not os.path.exists(CONFIG_PATH):
    print(f"[ОШИБКА] Конфиг {CONFIG_PATH} не найден!")
    sys.exit(1)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Окна
cv2.namedWindow("3D SCANNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("3D SCANNER", 50, 50)
cv2.resizeWindow("3D SCANNER", WINDOW_W, WINDOW_H)

cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)

# Камеры
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Функция для 3D визуализации
def show_3d_scan(profiles):
    """Показывает 3D облако точек с возможностью вращения"""
    print("[3D] Строим 3D визуализацию...")
    
    # Создаем сетку координат
    n_frames, n_points = profiles.shape
    
    # X - координата по ширине профиля (мм)
    x_coords = np.linspace(-50, 50, n_points)  # Примерный диапазон
    
    # Y - время (кадры)
    y_coords = np.linspace(0, RECORD_SECONDS, n_frames)
    
    # Z - высота (уже в мм)
    z_coords = profiles
    
    # Создаем 3D график
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Строим поверхность
    X, Y = np.meshgrid(x_coords, y_coords)
    surf = ax.plot_surface(X, Y, z_coords, cmap='jet', alpha=0.8, linewidth=0, antialiased=True)
    
    # Настройки
    ax.set_xlabel('Ширина (мм)')
    ax.set_ylabel('Время (с)')
    ax.set_zlabel('Высота (мм)')
    ax.set_title(f'3D Скан детали ({n_frames} кадров)')
    
    # Цветовая шкала
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Высота (мм)')
    
    # Автоматическое масштабирование
    ax.auto_scale_xyz([-60, 60], [0, RECORD_SECONDS], 
                      [np.min(z_coords)-2, np.max(z_coords)+2])
    
    plt.tight_layout()
    plt.show()
    
    print("[3D] Готово! Можешь вращать мышкой.")

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

# Переменные логики
base_disp = None
scale_mm_pix = 1.5
smoothed_dev = 0.0
history = []
status = "Нажми 'b' для калибровки нуля (стены)"

# Переменные для записи
is_recording = False
record_start_time = 0
recorded_profiles = []  # Список профилей (каждый профиль - массив z_mm)
recorded_timestamps = []

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    line_l = get_line_x(f1, config, 'l')
    line_r = get_line_x(f2, config, 'r')
    
    h, w = f1.shape[:2]
    current_disp = line_l - line_r
    raw_dev = 0.0
    plot_pts = []
    current_profile = []  # Для записи

    # Расчет дефектов
    for y in range(h):
        if line_l[y] > 0 and line_r[y] > 0:
            if base_disp is not None:
                delta = current_disp[y] - base_disp[y]
                z_mm = delta * scale_mm_pix
                current_profile.append(z_mm)  # Запоминаем для записи
                
                if abs(z_mm) > abs(raw_dev): 
                    raw_dev = z_mm
                
                px = int((y / h) * PANEL_W)
                py = int((GRAPH_H // 2) - (z_mm * 4.0))
                plot_pts.append((px, max(10, min(GRAPH_H-10, py))))

    # --- ЗАПИСЬ ПРОФИЛЕЙ ---
    if is_recording:
        elapsed = time.time() - record_start_time
        if elapsed < RECORD_SECONDS:
            if len(current_profile) > 10:  # Хотя бы 10 точек
                recorded_profiles.append(np.array(current_profile))
                recorded_timestamps.append(elapsed)
                status = f"ЗАПИСЬ: {elapsed:.1f}/{RECORD_SECONDS:.1f}с  [{len(recorded_profiles)} кадров]"
        else:
            # Останавливаем запись
            is_recording = False
            status = f"ЗАПИСЬ ЗАВЕРШЕНА! Сохранено {len(recorded_profiles)} кадров"
            print(f"[ЗАПИСЬ] Сохранено {len(recorded_profiles)} профилей за {RECORD_SECONDS}с")
            
            # Сохраняем в npy
            if len(recorded_profiles) > 0:
                # Приводим к единой длине (берем минимальную)
                min_len = min([len(p) for p in recorded_profiles])
                if min_len > 10:
                    # Обрезаем все профили до минимальной длины
                    profiles_trimmed = np.array([p[:min_len] for p in recorded_profiles])
                    # Сохраняем
                    np.save("scan_3d.npy", profiles_trimmed)
                    print(f"[СОХРАНЕНО] scan_3d.npy shape: {profiles_trimmed.shape}")
                    
                    # Показываем 3D визуализацию
                    show_3d_scan(profiles_trimmed)
                else:
                    status = "ОШИБКА: слишком мало точек в профилях!"
                    print("[ОШИБКА] Профили слишком короткие!")

    # --- КРАСИВЫЙ ГРАФИК ---
    plot_img = np.zeros((GRAPH_H, PANEL_W, 3), dtype=np.uint8)
    plot_img.fill(25)
    
    # Сетка
    for y in range(0, GRAPH_H, 40):
        cv2.line(plot_img, (0, y), (PANEL_W, y), (45, 45, 45), 1)
    for x in range(0, PANEL_W, 60):
        cv2.line(plot_img, (x, 0), (x, GRAPH_H), (45, 45, 45), 1)
    
    # Нулевая линия
    cv2.line(plot_img, (0, GRAPH_H//2), (PANEL_W, GRAPH_H//2), (100, 100, 100), 2)
    
    # Зона допуска
    tol_pixels = int(TOLERANCE_MM * 4.0)
    overlay = plot_img.copy()
    cv2.rectangle(overlay, (0, GRAPH_H//2 - tol_pixels), (PANEL_W, GRAPH_H//2 + tol_pixels), (0, 80, 0), -1)
    cv2.addWeighted(overlay, 0.2, plot_img, 0.8, 0, plot_img)
    
    # Основной профиль
    if len(plot_pts) > 1:
        color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
        cv2.polylines(plot_img, [np.array(plot_pts)], False, color, 3, cv2.LINE_AA)
        fill_pts = np.array(plot_pts + [(PANEL_W-1, GRAPH_H//2), (0, GRAPH_H//2)])
        cv2.fillPoly(plot_img, [fill_pts], (color[0]//4, color[1]//4, color[2]//4))
    
    # Индикатор записи (красная рамка)
    if is_recording:
        cv2.rectangle(plot_img, (0, 0), (PANEL_W-1, GRAPH_H-1), (0, 0, 255), 3)
        # Красная точка в углу
        cv2.circle(plot_img, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(plot_img, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # История пиков
    for i, p in enumerate(history):
        py = int((GRAPH_H // 2) - (p * 4.0))
        py = max(10, min(GRAPH_H-10, py))
        if i < PANEL_W:
            cv2.circle(plot_img, (i, py), 2, (0, 165, 255), -1)
    
    # Подписи
    cv2.putText(plot_img, f"+{TOLERANCE_MM:.1f}mm", (10, GRAPH_H//2 - tol_pixels - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(plot_img, f"-{TOLERANCE_MM:.1f}mm", (10, GRAPH_H//2 + tol_pixels + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    # Текущее значение
    val_color = (0, 255, 0) if abs(smoothed_dev) < TOLERANCE_MM else (0, 0, 255)
    cv2.putText(plot_img, f"{smoothed_dev:+.2f} mm", (PANEL_W-180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, val_color, 3)
    
    cv2.putText(plot_img, f"Scale: {scale_mm_pix:.3f} mm/px", (PANEL_W-180, GRAPH_H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(plot_img, f"Points: {len(plot_pts)}", (10, GRAPH_H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # История пиков
    if base_disp is not None and abs(raw_dev) > 0.1:
        history.append(raw_dev)
        if len(history) > PANEL_W: history.pop(0)

    # Сглаживание
    smoothed_dev = (SMOOTH_FACTOR * raw_dev) + ((1.0 - SMOOTH_FACTOR) * smoothed_dev)
    
    # Видео панель
    vis_l = cv2.resize(f1, (PANEL_W//2, VIDEO_H))
    vis_r = cv2.resize(f2, (PANEL_W//2, VIDEO_H))
    video_panel = np.hstack((vis_l, vis_r))
    
    # Статусная строка
    status_bar = np.zeros((STATUS_H, PANEL_W, 3), dtype=np.uint8)
    status_bar.fill(20)
    cv2.putText(status_bar, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    color = (0, 255, 0) if base_disp is not None else (0, 0, 255)
    cv2.circle(status_bar, (PANEL_W-40, 25), 10, color, -1)
    
    # Индикатор записи в статусе
    if is_recording:
        cv2.circle(status_bar, (PANEL_W-80, 25), 8, (0, 0, 255), -1)
    
    # Сборка
    top = np.vstack((video_panel, status_bar))
    full = np.vstack((top, plot_img))
    cv2.rectangle(full, (0, 0), (full.shape[1]-1, full.shape[0]-1), (60, 60, 60), 2)
    
    cv2.imshow("3D SCANNER", full)
    cv2.imshow("PROJECTOR", pattern)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('b'):
        if np.count_nonzero(current_disp > 0) > 100:
            base_disp = current_disp.copy()
            history.clear()
            smoothed_dev = 0.0
            status = "База установлена (0 мм)"
            print("[OK] База захвачена")
        else:
            status = "ОШИБКА: Линия не найдена!"
    elif key == ord('r'):
        if base_disp is None:
            status = "СНАЧАЛА НАЖМИ 'b' ДЛЯ КАЛИБРОВКИ!"
        elif is_recording:
            status = "ЗАПИСЬ УЖЕ ИДЕТ!"
        else:
            # Старт записи
            is_recording = True
            record_start_time = time.time()
            recorded_profiles = []
            recorded_timestamps = []
            status = f"ЗАПИСЬ СТАРТ! {RECORD_SECONDS}с..."
            print("[ЗАПИСЬ] Начинаем запись... Двигай деталь!")
    elif key == ord('c'):
        if base_disp is not None and abs(smoothed_dev) > 0.1:
            scale_mm_pix = (REAL_BLOCK_MM / abs(smoothed_dev)) * scale_mm_pix
            status = f"Калибровка: {scale_mm_pix:.4f} мм/пикс"
            print(f"[КАЛИБРОВКА] Новый масштаб: {scale_mm_pix:.4f}")
        else: 
            status = "Ошибка калибровки (нужен брусок)"

cap1.release()
cap2.release()
cv2.destroyAllWindows()

