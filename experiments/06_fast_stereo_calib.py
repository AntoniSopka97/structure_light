import os
import cv2
import numpy as np
import glob
import sys

# Настройки твоей шахматной доски
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 26.0
NPZ_PATH = "stereo_calibration.npz"

print("=== ЗАПУСК ПУЛЕНЕПРОБИВАЕМОЙ СТЕРЕО-КАЛИБРОВКИ ===")

# Было:
# images_l = sorted(glob.glob('calibration_data/left/*.png'))
# images_r = sorted(glob.glob('calibration_data/right/*.png'))

# Стало (Исправляем перепутанные провода камер):
images_l = sorted(glob.glob('calibration_data/right/*.png')) # Читаем ПРАВУЮ папку как ЛЕВЫЙ глаз
images_r = sorted(glob.glob('calibration_data/left/*.png'))  # Читаем ЛЕВУЮ папку как ПРАВЫЙ глаз


if len(images_l) == 0 or len(images_r) == 0:
    print("[ОШИБКА] Кадры калибровки в папке calibration_data/ не найдены!")
    sys.exit(1)

print(f"[ИНФО] Найдено {len(images_l)} пар кадров.")

# Критерии субпиксельной точности
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Базовая 3D сетка углов доски в мм
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

all_objpoints = []  
all_imgpoints_l = [] 
all_imgpoints_r = [] 
img_shape = None

print("Идет точный поиск субпиксельных углов...")

for img_l_path, img_r_path in zip(images_l, images_r):
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = gray_l.shape[::-1]
        
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)
    
    if ret_l and ret_r:
        # Максимально точный поиск субпикселей
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        
        all_objpoints.append(objp)
        all_imgpoints_l.append(corners_l2)
        all_imgpoints_r.append(corners_r2)

print(f"[ИНФО] Углы верифицированы на {len(all_objpoints)} парах кадров.")

# Инициализируем пустые матрицы камер
mtx_l = np.zeros((3, 3))
dist_l = np.zeros((1, 5))
mtx_r = np.zeros((3, 3))
dist_r = np.zeros((1, 5))

print("Запуск сквозной совместной стерео-калибровки...")
# Флаги: Считаем интринсики с нуля, фиксируем отсутствие тангенциальной дисторсии для стабильности
flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3

ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    all_objpoints, all_imgpoints_l, all_imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r, img_shape, 
    criteria=criteria, flags=flags
)

print(f"[УСПЕХ] Калибровка завершена! Новая ошибка репроекции: {ret:.4f} пикселей.")

if ret > 2.0:
    print("[ВНИМАНИЕ] Ошибка всё еще высока. Возможно, перепутаны левые и правые папки!")

# Расчет виртуального выравнивания (Stereo Rectify)
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T, alpha=0
)

# Строим карты трансформации пикселей
map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_32FC1)
map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_32FC1)

# Сохраняем финальный npz
np.savez(NPZ_PATH, 
         map_l_x=map_l_x, map_l_y=map_l_y,
         map_r_x=map_r_x, map_r_y=map_r_y, Q=Q)

print(f"[ГОТОВО] Свежий {NPZ_PATH} записан и очищен от ошибок!")
