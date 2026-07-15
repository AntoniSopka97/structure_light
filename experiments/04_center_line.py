import cv2
import numpy as np
import sys
import json
import os

PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"Threshold": 50, "Min_Intensity": 30}

config = load_config()
thresh_val = config["Threshold"]
min_intensity = config["Min_Intensity"]

print(f"Загружены пороги: Threshold={thresh_val}, Min_Intensity={min_intensity}")

# Настройка окон на основном мониторе
cv2.namedWindow("Webcam LEFT (1)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam LEFT (1)", 50, 50)
cv2.namedWindow("Webcam RIGHT (0)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam RIGHT (0)", 750, 50)

# Настройка проектора (X=3440)
cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Маска для проектора
pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)

# Инициализация вебок
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def find_line_center_vectorized(frame, thresh, min_int):
    """
    Быстрый векторизованный субпиксельный поиск центра линии (Центроид) через NumPy.
    Без медленных циклов Python.
    """
    # Выдергиваем синий канал (индекс 0 в BGR)
    b_channel = frame[:, :, 0]
    
    # Срезаем шум по порогу Threshold (все что ниже -> в 0)
    _, masked = cv2.threshold(b_channel, thresh, 255, cv2.THRESH_TOZERO)
    masked = masked.astype(np.float32)
    
    # Считаем сумму интенсивностей по каждому столбцу (ось 0)
    sum_intensity = np.sum(masked, axis=0)
    
    # Находим максимальную яркость в каждом столбце
    max_intensity = np.max(masked, axis=0)
    
    # Строим сетку индексов строк [0, 1, 2 ... H-1] в виде вертикального вектора
    h, w = masked.shape
    y_indices = np.arange(h, dtype=np.float32).reshape(-1, 1)
    
    # Скалярное умножение индексов на матрицу яркостей (каждый столбец умножается на y_indices)
    sum_y_intensity = np.sum(masked * y_indices, axis=0)
    
    # Задаем базовый массив дефолтных значений (-1.0 означает линии нет)
    centers_y = np.full(w, -1.0, dtype=np.float32)
    
    # Маска валидных столбцов: где есть пиксель ярче Min_Intensity И сумма > 0
    valid_cols = (max_intensity >= min_int) & (sum_intensity > 0)
    
    # Считаем центроид только там, где маска истинна
    centers_y[valid_cols] = sum_y_intensity[valid_cols] / sum_intensity[valid_cols]
    
    return centers_y

print("Векторизованный субпиксельный детектор запущен. Жми 'q' для выхода.")

while True:
    cv2.imshow("PROJECTOR", pattern)
    
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    
    if ret1 and ret2:
        # Считаем координаты центров
        centers1 = find_line_center_vectorized(f1, thresh_val, min_intensity)
        centers2 = find_line_center_vectorized(f2, thresh_val, min_intensity)
        
        vis1 = f1.copy()
        vis2 = f2.copy()
        
        # Отрисовка красных точек на ЛЕВОЙ камере (1)
        for x in range(f1.shape[1]):
            y1 = centers1[x]
            if y1 > 0:
                cv2.circle(vis1, (x, int(y1)), 3, (0, 0, 255), -1) # Сплошная точка
                
        # Отрисовка красных точек на ПРАВОЙ камере (0)
        for x in range(f2.shape[1]):
            y2 = centers2[x]
            if y2 > 0:
                cv2.circle(vis2, (x, int(y2)), 3, (0, 0, 255), -1) # Сплошная точка на vis2!

        # Выводим картинки на экран
        cv2.imshow("Webcam LEFT (1)", cv2.resize(vis1, (640, 360)))
        cv2.imshow("Webcam RIGHT (0)", cv2.resize(vis2, (640, 360)))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
