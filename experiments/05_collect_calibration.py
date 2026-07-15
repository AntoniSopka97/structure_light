import cv2
import os
import sys

# Параметры твоей доски
CHECKERBOARD = (9, 6)

# Создаем папки для кадров, если их еще нет
PATH_LEFT = "calibration_data/left"
PATH_RIGHT = "calibration_data/right"
os.makedirs(PATH_LEFT, exist_ok=True)
os.makedirs(PATH_RIGHT, exist_ok=True)

# Твои рабочие индексы на Linux (Левая - 1, Правая - 0)
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap1.isOpened() or not cap2.isOpened():
    print("Ошибка! Не удалось открыть камеры.")
    sys.exit(1)

print("--- СБОР КАДРОВ СТЕРЕО-КАЛИБРОВКИ (9х6) ---")
print("Выключи синюю линию проектора, включи свет в комнате.")
print("Показывай шахматную доску ОБЕИМ камерам ОДНОВРЕМЕННО.")
print("Нажми ПРОБЕЛ или 'c' — чтобы сохранить пару кадров.")
print("Нажми 'q' — выход.")

img_counter = 0

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    
    if not ret1 or not ret2:
        continue
        
    vis1 = f1.copy()
    vis2 = f2.copy()
    
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    # Поиск углов для индикации на экране
    ret_find1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    ret_find2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)
    
    if ret_find1:
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, ret_find1)
    if ret_find2:
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, ret_find2)
        
    # Показываем окна на основном мониторе
    cv2.imshow("Webcam LEFT (1) - Press Space to Save", cv2.resize(vis1, (1920, 1080)))
    cv2.imshow("Webcam RIGHT (0) - Press Space to Save", cv2.resize(vis2, (1920, 1080)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') or key == ord(' '):
        # Нам критически важно, чтобы доска была видна на ОБЕИХ камерах одновременно!
        if ret_find1 and ret_find2:
            img_name_l = os.path.join(PATH_LEFT, f"frame_{img_counter:03d}.png")
            img_name_r = os.path.join(PATH_RIGHT, f"frame_{img_counter:03d}.png")
            
            # Сохраняем чистые оригиналы без цветных сеток
            cv2.imwrite(img_name_l, f1)
            cv2.imwrite(img_name_r, f2)
            
            print(f"[OK] Пара кадров #{img_counter} успешно сохранена!")
            img_counter += 1
        else:
            print(f"[ВНИМАНИЕ] Доска должна быть видна на ОБЕИХ камерах! Левая: {ret_find1}, Правая: {ret_find2}")

cap1.release()
cap2.release()
cv2.destroyAllWindows()
