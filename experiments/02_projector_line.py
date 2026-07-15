import cv2
import numpy as np
import sys
import json
import os

PROJ_W, PROJ_H = 1920, 1080
CONFIG_PATH = "config.json"

# 1. Функция загрузки конфига
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                print("--- Конфиг успешно загружен из файла ---")
                return json.load(f)
        except Exception as e:
            print(f"Ошибка чтения конфига: {e}")
    # Дефолтные значения, если файла нет
    return {"Threshold": 50, "Min_Intensity": 30}

# 2. Функция сохранения конфига
def save_config(thresh, min_int):
    config = {"Threshold": thresh, "Min_Intensity": min_int}
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        print(f"--- Настройки успешно сохранены в {CONFIG_PATH}! ---")
    except Exception as e:
        print(f"Не удалось сохранить конфиг: {e}")

def nothing(x):
    pass

# Загружаем сохраненные ранее или дефолтные параметры
saved_config = load_config()

# 3. Создаем окно настроек "TUNER" на главном мониторе
cv2.namedWindow("TUNER", cv2.WINDOW_NORMAL)
cv2.moveWindow("TUNER", 50, 550)
cv2.resizeWindow("TUNER", 400, 200)

# Привязываем бегунки и выставляем им значения из конфига
cv2.createTrackbar("Threshold", "TUNER", saved_config["Threshold"], 255, nothing)
cv2.createTrackbar("Min_Intensity", "TUNER", saved_config["Min_Intensity"], 255, nothing)

# 4. Настраиваем окна превью вебок на основном мониторе
cv2.namedWindow("Webcam LEFT (1)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam LEFT (1)", 50, 50)

cv2.namedWindow("Webcam RIGHT (0)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam RIGHT (0)", 750, 50)

# 5. Настраиваем окно проектора (улетает на X=3440)
cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.moveWindow("PROJECTOR", 3440, 0)
cv2.imshow("PROJECTOR", np.zeros((300, 300, 3), dtype=np.uint8))
cv2.waitKey(100)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Маска с синей линией для проектора
pattern = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
cv2.line(pattern, (PROJ_W // 2, 0), (PROJ_W // 2, PROJ_H), (255, 0, 0), 6)

# 6. Инициализация вебок (твои индексы 1 и 0)
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap1.isOpened() or not cap2.isOpened():
    print("Ошибка вебок!")
    sys.exit(1)

print("Тюнер запущен. 's' - сохранить настройки, 'q' - выход.")

while True:
    cv2.imshow("PROJECTOR", pattern)
    
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    
    if ret1 and ret2:
        thresh_val = cv2.getTrackbarPos("Threshold", "TUNER")
        min_int = cv2.getTrackbarPos("Min_Intensity", "TUNER")
        
        # Обработка левой камеры (выделяем синий канал)
        b_channel1 = f1[:, :, 0]
        _, mask1 = cv2.threshold(b_channel1, thresh_val, 255, cv2.THRESH_BINARY)
        contour_f1 = f1.copy()
        contour_f1[mask1 > 0] = (0, 255, 0)
        
        # Обработка правой камеры
        b_channel2 = f2[:, :, 0]
        _, mask2 = cv2.threshold(b_channel2, thresh_val, 255, cv2.THRESH_BINARY)
        contour_f2 = f2.copy()
        contour_f2[mask2 > 0] = (0, 255, 0)

        cv2.imshow("Webcam LEFT (1)", cv2.resize(contour_f1, (640, 360)))
        cv2.imshow("Webcam RIGHT (0)", cv2.resize(contour_f2, (640, 360)))
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Ловим текущие позиции ползунков и пишем в файл
        save_config(thresh_val, min_int)

cap1.release()
cap2.release()
cv2.destroyAllWindows()
