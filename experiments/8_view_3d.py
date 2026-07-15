import numpy as np
import cv2
import sys
import os

print("=== ГЕНЕРАЦИЯ ОЧИЩЕННОЙ ТЕПЛОВОЙ КАРТЫ ЛИСТА ===")

PLY_PATH = "stereo_scan_result.ply"

if not os.path.exists(PLY_PATH):
    print(f"[ОШИБКА] Файл {PLY_PATH} не найден. Сначала сделай сканирование!")
    sys.exit(1)

# 1. Читаем координаты из PLY файла
points = []
with open(PLY_PATH, "r") as f:
    lines = f.readlines()
    start_reading = False
    for line in lines:
        if "end_header" in line:
            start_reading = True
            continue
        if start_reading:
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])

pts = np.array(points)
if len(pts) == 0:
    print("[ОШИБКА] Облако точек пустое!")
    sys.exit(1)

# Выдергиваем координаты
X = pts[:, 0]
Y = pts[:, 1]
Z = pts[:, 2]

# 2. Переводим облако точек в регулярную плотную 2D матрицу (картинку)
# Определяем размеры сетки на основе записанных данных
x_min, x_max = int(np.min(X)), int(np.max(X))
y_min, y_max = int(np.min(Y)), int(np.max(Y))

# Создаем пустую матрицу высот
grid_w = x_max - x_min + 1
grid_h = y_max - y_min + 1

# Защита от слишком огромной или пустой матрицы
if grid_w <= 0 or grid_h <= 0 or grid_w > 4000 or grid_h > 10000:
    print("[ИНФО] Координаты X/Y сырые. Авто-масштабирование сетки...")
    grid_w, grid_h = 800, 600
    # Принудительно масштабируем координаты в пиксели матрицы
    x_indices = np.interp(X, (X.min(), X.max()), (0, grid_w - 1)).astype(np.int32)
    y_indices = np.interp(Y, (Y.min(), Y.max()), (0, grid_h - 1)).astype(np.int32)
else:
    x_indices = (X - x_min).astype(np.int32)
    y_indices = (Y - y_min).astype(np.int32)

height_matrix = np.zeros((grid_h, grid_w), dtype=np.float32)
height_matrix[y_indices, x_indices] = Z

# 3. ПРОМЫШЛЕННАЯ ФИЛЬТРАЦИЯ И ОЧИСТКА ОТ ШУМА ШВАБРЫ
# А. Применяем медианный фильтр ядра 5х5, чтобы намертво срезать одиночные выстрелы в космос
height_filtered = cv2.medianBlur(height_matrix, 5)

# Б. Вводим зону нечувствительности (Dead Zone): всё, что ниже 3 мм перепада — это цеховой шум вебок
# Оставляем только то, что реально возвышается над стеной
dead_zone_mm = 3.0
height_filtered[np.abs(height_filtered) < dead_zone_mm] = 0.0

# 4. РЕНДЕРИНГ ТЕПЛОВОЙ КАРТЫ ДЛЯ ОПЕРАТОРА ПРЕССА
# Нормализуем высоты в диапазон 0-255 для отрисовки цвета
z_max_val = np.max(np.abs(height_filtered))
if z_max_val == 0: z_max_val = 1.0 # защита от деления на ноль

# Переводим матрицу в байтовый формат uint8
img_gray = np.zeros_like(height_filtered, dtype=np.uint8)

# Разделяем горбы и ямы: горбы сделаем яркими
img_gray[height_filtered > 0] = ((height_filtered[height_filtered > 0] / z_max_val) * 255).astype(np.uint8)

# Применяем цветовую палитру JET (Синий -> Зеленый -> Красный)
# На идеально ровных участках (где мы занулили шум) цвет будет зеленым/синеватым.
# Где прошла швабра (настоящий горб) — вспыхнет ярко-красное плотное пятно!
heatmap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

# Если на ровных участках цвет уплыл в синеву, принудительно закрасим абсолютный ноль в зеленый (нейтральный) цвет
heatmap[height_filtered == 0.0] = [0, 180, 0] # Спокойный зеленый цвет нормы

# 5. Выводим плоскую карту листа на твой ультраширокий экран
cv2.namedWindow("PROMETALL 3D - OPERATOR HEATMAP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PROMETALL 3D - OPERATOR HEATMAP", 1200, 700)

# Добавим разметку максимального дефекта текстом
cv2.putText(heatmap, f"MAX DEFECT HEIGHT: {z_max_val:.1f} mm", (30, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.putText(heatmap, "GREEN = OK (IN TOLERANCE) | RED = HUMP (NEED PRESS)", (30, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

print("Карта отрендерена! Жми 'q' в окне картинки для выхода.")
while True:
    cv2.imshow("PROMETALL 3D - OPERATOR HEATMAP", heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
