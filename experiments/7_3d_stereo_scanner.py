import cv2
import numpy as np
import json
import os
import sys

PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"

print("=== ЗАПУСК ПУЛЕНЕПРОБИВАЕМОГО 3D СТЕРЕО-СКАНЕРА ДЛЯ СТАЛИ ===")

# 1. Загрузка твоих идеальных порогов и залоченных ROI
if not os.path.exists(CONFIG_PATH):
    print(f"[ОШИБКА] Конфиг {CONFIG_PATH} не найден!")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

thresh_val = config["Threshold"]
min_intensity = config["Min_Intensity"]
roi_l_start = config["roi_l_start"]
roi_l_end = config["roi_l_end"]
roi_r_start = config["roi_r_start"]
roi_r_end = config["roi_r_end"]

print(f"[ОК] Коридоры загружены -> Левый: [{roi_l_start}:{roi_l_end}], Правый: [{roi_r_start}:{roi_r_end}]")

# 2. Настройка окон интерфейса оператора пресса (Экран 3440х1440)
cv2.namedWindow("3D SCANNER - CONTROL INTERFACE", cv2.WINDOW_NORMAL)
cv2.moveWindow("3D SCANNER - CONTROL INTERFACE", 50, 50)
cv2.resizeWindow("3D SCANNER - CONTROL INTERFACE", 1300, 480)

# Настройка проектора (Улетает на правый экран X=3440)
cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Маска с синей линией для проектора
pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)

# 3. Инициализация вебок (твои индексы 1 и 0)
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



def render_and_filter_heatmap(points_3d):
    """Превращает 3D облако в чистую отфильтрованную 2D тепловую карту листа"""
    pts = np.array(points_3d)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Задаем фиксированную сетку под наш монитор оператора пресса (800х600 пикселей)
    grid_w, grid_h = 800, 600
    x_indices = np.interp(X, (X.min(), X.max()), (0, grid_w - 1)).astype(np.int32)
    y_indices = np.interp(Y, (Y.min(), Y.max()), (0, grid_h - 1)).astype(np.int32)
    
    height_matrix = np.zeros((grid_h, grid_w), dtype=np.float32)
    height_matrix[y_indices, x_indices] = Z
    
    # --- ВАШИ ИСПРАВЛЕНИЯ (ФИЛЬТРАЦИЯ ШУМА) ---
    # Фильтр 1: Срезаем одиночный импульсный шум и выстрелы вебок в космос
    height_filtered = cv2.medianBlur(height_matrix, 5)
    
    # Фильтр 2: Мертвая зона (Dead Zone). Всё, что меньше 3 мм перепада — цеховой шум, гасим в 0
    dead_zone_mm = 3.0
    height_filtered[np.abs(height_filtered) < dead_zone_mm] = 0.0
    # ----------------------------------------
    
    z_max_val = np.max(np.abs(height_filtered))
    if z_max_val == 0: z_max_val = 1.0
    
    # Переводим в байты для раскраски
    img_gray = np.zeros_like(height_filtered, dtype=np.uint8)
    img_gray[height_filtered > 0] = ((height_filtered[height_filtered > 0] / z_max_val) * 255).astype(np.uint8)
    
    # Накладываем палитру JET (Синий -> Зеленый -> Красный)
    heatmap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    
    # ТВОЯ ЛОГИКА: закрашиваем идеальную норму (0.0 мм) в чистый, ровный зеленый цвет
    heatmap[height_filtered == 0.0] = (0, 180, 0)
    
    # Добавляем текстовые маркеры дефекта
    cv2.putText(heatmap, f"MAX DEFECT: {z_max_val:.1f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(heatmap, "GREEN = OK | RED = HUMP (NEED PRESS)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Показываем плоскую отфильтрованную карту оператору
    cv2.namedWindow("PROMETALL 3D - OPERATOR HEATMAP", cv2.WINDOW_NORMAL)
    cv2.imshow("PROMETALL 3D - OPERATOR HEATMAP", heatmap)
    print(f"[ПАНЕЛЬ] Карта высот выведена на экран. Максимальный горб: {z_max_val:.1f} мм.")

def get_line_x_profile_raw_roi(frame, thresh, min_int, r_start, r_end):
    """Ищем субпиксельный центр масс линии строго внутри твоего залоченного JSON коридора"""
    b_channel = frame[:, :, 0]
    
    # Зануляем всё, что вне твоего ROI
    roi_masked = np.zeros_like(b_channel)
    roi_masked[:, r_start:r_end] = b_channel[:, r_start:r_end]
    
    _, masked = cv2.threshold(roi_masked, thresh, 255, cv2.THRESH_TOZERO)
    masked = masked.astype(np.float32)
    
    sum_intensity = np.sum(masked, axis=1)
    max_intensity = np.max(masked, axis=1)
    h, w = masked.shape
    x_indices = np.arange(w, dtype=np.float32).reshape(1, -1)
    
    sum_x_intensity = np.sum(masked * x_indices, axis=1)
    
    centers_x = np.full(h, -1.0, dtype=np.float32)
    valid_rows = (max_intensity >= min_int) & (sum_intensity > 0)
    centers_x[valid_rows] = sum_x_intensity[valid_rows] / sum_intensity[valid_rows]
    return centers_x

def save_to_ply(filename, points_3d):
    """Сохранение облака точек в формат PLY с цветовым кодированием рельефа"""
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        for p in points_3d:
            x_v, y_v, z_v = p[0], p[1], p[2]
            # Градиент: горбы краснеют, ямы синеют
            r = int(np.clip(abs(z_v) * 25, 0, 255)) if z_v > 0 else 0
            b = int(np.clip(abs(z_v) * 25, 0, 255)) if z_v <= 0 else 0
            g = int(max(0, 255 - max(r, b)))
            f.write(f"{x_v:.2f} {y_v:.2f} {z_v:.2f} {r} {g} {b}\n")
    print(f"[УСПЕХ] Физическая 3D карта высот детали записана: {filename}")

base_wall_disparity = None # Базовая диспаратность пустой плоской стены
scanned_cloud_3d = []      # Хранилище точек рельефа
is_scanning = False
frame_counter = 0
status_text = "STATUS: SCENE ACTIVE. Press 'b' to reset baseline (WALL)."

while True:
    cv2.imshow("PROJECTOR", pattern)
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue
    
    # Ищем линии СТРОГО на сырых кадрах внутри твоих коридоров из json
    line_x_left = get_line_x_profile_raw_roi(f1, thresh_val, min_intensity, roi_l_start, roi_l_end)
    line_x_right = get_line_x_profile_raw_roi(f2, thresh_val, min_intensity, roi_r_start, roi_r_end)
    
    h, w = f1.shape[:2]
    current_frame_disparity = np.zeros(h, dtype=np.float32)
    current_frame_points_3d = []
    
    vis_l, vis_r = f1.copy(), f2.copy()
    
    # Отрисовываем твои залоченные коридоры ROI тонкими линиями на экране
    cv2.line(vis_l, (roi_l_start, 0), (roi_l_start, h), (0, 255, 255), 1)
    cv2.line(vis_l, (roi_l_end, 0), (roi_l_end, h), (0, 255, 255), 1)
    cv2.line(vis_r, (roi_r_start, 0), (roi_r_start, h), (0, 255, 255), 1)
    cv2.line(vis_r, (roi_r_end, 0), (roi_r_end, h), (0, 255, 255), 1)
    
    for y in range(h):
        xl, xr = line_x_left[y], line_x_right[y]
        if xl > 0 and xr > 0:
            # Подсвечиваем центроид линии красной точкой — ты увидишь её СТРОГО по центру ROI
            cv2.circle(vis_l, (int(xl), y), 2, (0, 0, 255), -1)
            cv2.circle(vis_r, (int(xr), y), 2, (0, 0, 255), -1)
            
            # Чистый сдвиг луча между камерами в пикселях
            current_frame_disparity[y] = xl - xr
            
            # Если идет сканирование детали и базовый профиль стены сохранен
            if is_scanning and base_wall_disparity is not None:
                base_disp = base_wall_disparity[y]
                if base_disp != 0:
                    # Разница диспаратности: чем ближе горб, тем сильнее сдвиг пикселей луча!
                    delta_disp = (xl - xr) - base_disp
                    
                    # Эмпирический перевод пиксельного излома в мм высоты для настольного теста
                    # 1 пиксель сдвига примерно равен 1.5 мм высоты детали на столе
                    z_height_mm = delta_disp * 1.5 
                    
                    # Продольный шаг конвейера во времени (2 мм на кадр)
                    time_step_y = float(frame_counter) * 2.0
                    # Поперечная координата X (просто пиксели, переведенные в масштаб мм)
                    x_mm = (xl - w // 2) * 0.5
                    
                    current_frame_points_3d.append([x_mm, time_step_y, z_height_mm])

    if is_scanning and len(current_frame_points_3d) > 0:
        scanned_cloud_3d.extend(current_frame_points_3d)
        frame_counter += 1

    # Вывод интерфейса оператора
    p_l = cv2.resize(vis_l, (640, 360))
    p_r = cv2.resize(vis_r, (640, 360))
    interface_img = np.hstack((p_l, p_r))
    
     
    status_bar = np.zeros((70, interface_img.shape[1], 3), dtype=np.uint8)

    color = (0, 0, 255) if is_scanning else (0, 255, 0)
    cv2.putText(status_bar, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    interface_img = np.vstack((interface_img, status_bar))
    
    cv2.imshow("3D SCANNER - CONTROL INTERFACE", interface_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        # Кнопка 'b' — Запоминаем плоскую пустую стену как Ноль
        if np.any(current_frame_disparity != 0):
            base_wall_disparity = current_frame_disparity.copy()
            status_text = "STATUS: BASE WALL CALIBRATED. Press 's' to START recording."
            print("[БАЗА ОК] Профиль пустой стены успешно сохранен.")
        else:
            status_text = "STATUS: ERROR. Cannot see line inside ROI corridors!"
            print("[ОШИБКА] Линия пустая, проверь маску.")
    elif key == ord('s'):
        if base_wall_disparity is None:
            status_text = "STATUS: ERROR. Press 'b' first to calibrate empty wall!"
        else:
            if not is_scanning:
                is_scanning = True
                scanned_cloud_3d = []
                frame_counter = 0
                status_text = "STATUS: RECORDING ACTIVE! Move your board through the beam..."
                print("[ЗАПИСЬ ЗАПУЩЕНА] Копим кадры рельефа.")
            else:
                is_scanning = False
                status_text = "STATUS: COMPUTING MAP... Saving PLY."
                cv2.imshow("3D SCANNER - CONTROL INTERFACE", interface_img)
                cv2.waitKey(50)
                
                if len(scanned_cloud_3d) > 0:
                    filename = "stereo_scan_result.ply"
                    save_to_ply(filename, scanned_cloud_3d)
                    
                    # ЗАПУСКАЕМ НАШУ МАГИЮ: Очищаем шум и рендерим плоскую теплокарту!
                    render_and_filter_heatmap(scanned_cloud_3d)
                    
                    status_text = f"STATUS: MAP SAVED to {filename}. Heatmap active."
                else:
                    status_text = "STATUS: ERROR. Empty cloud."

cap1.release()
cap2.release()
cv2.destroyAllWindows()
