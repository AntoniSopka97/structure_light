import numpy as np
import open3d as o3d
import sys

def load_and_view_scan(filename="scan_3d.npy"):
    """
    Загружает npy и показывает 3D скан с ПРАВИЛЬНОЙ ОРИЕНТАЦИЕЙ:
    - X (красная) → вправо
    - Y (зеленая) → вверх
    - Z (синяя) → на оператора (глубина)
    """
    
    print("[3D] Загрузка файла...")
    
    try:
        profiles = np.load(filename)
        print(f"[ЗАГРУЗКА] {filename} shape: {profiles.shape}")
    except FileNotFoundError:
        print(f"[ОШИБКА] Файл {filename} не найден!")
        return
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить: {e}")
        return
    
    n_frames, n_points = profiles.shape
    print(f"[OK] Кадров: {n_frames}, Точек в профиле: {n_points}")
    
    # --- РЕАЛЬНЫЕ РАЗМЕРЫ ---
    REAL_LENGTH_MM = 400.0  # Длина скана (по вертикали лазера)
    REAL_WIDTH_MM = 100.0   # Ширина профиля
    HEIGHT_SCALE = 3.0      # Усиление рельефа
    
    print(f"[OK] Реальная длина скана: {REAL_LENGTH_MM} мм")
    print("[OK] Ориентация: X→вправо, Y→вверх, Z→на оператора")
    
    # Создаем координаты в СИСТЕМЕ КООРДИНАТ ЛАЗЕРА (вертикальный луч)
    # В кадре: строки (y) идут сверху вниз, столбцы (x) слева направо
    x_coords = np.linspace(-REAL_WIDTH_MM/2, REAL_WIDTH_MM/2, n_points)  # ширина
    y_coords = np.linspace(0, REAL_LENGTH_MM, n_frames)                   # длина по лучу
    
    points = []
    colors = []
    
    z_min, z_max = np.min(profiles), np.max(profiles)
    z_range = z_max - z_min if z_max > z_min else 1.0
    print(f"[OK] Исходный диапазон высот: {z_min:.2f} .. {z_max:.2f} мм")
    
    print("[3D] Строю облако точек с правильной ориентацией...")
    
    for i in range(n_frames):
        for j in range(n_points):
            # Исходные данные
            x = x_coords[j]                     # по горизонтали (ширина)
            y = y_coords[i]                     # по вертикали (длина скана)
            z = profiles[i, j] * HEIGHT_SCALE   # рельеф (высота/глубина)
            
            # --- ПРАВИЛЬНОЕ ПРЕОБРАЗОВАНИЕ КООРДИНАТ ---
            # Мы хотим: X→вправо, Y→вверх, Z→на оператора (глубина)
            # В системе координат камеры: Y вверх, X вправо, Z вперед/назад
            # В нашем случае: рельеф (z) должен идти по глубине (Z), 
            # а длина скана (y) идет вверх (Y)
            points.append([x, y, z])  # <-- ТАК МЫ СОХРАНЯЕМ Y ВВЕРХ, А НЕ МЕНЯЕМ МЕСТАМИ
            
            # Цвет по рельефу (z) — насыщенный
            norm_z = (z - z_min * HEIGHT_SCALE) / (z_range * HEIGHT_SCALE)
            norm_z = np.clip(norm_z, 0, 1)
            
            # Jet colormap
            if norm_z < 0.5:
                r = 0
                g = int(255 * norm_z / 0.5)
                b = 255
            else:
                r = int(255 * (norm_z - 0.5) / 0.5)
                g = int(255 * (1 - (norm_z - 0.5) / 0.5))
                b = 0
            colors.append([r/255, g/255, b/255])
    
    if len(points) == 0:
        print("[ОШИБКА] Нет валидных точек!")
        return
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"[OK] Создано {len(points)} точек")
    print(f"[OK] Диапазон X (вправо): {np.min(points[:,0]):.1f}..{np.max(points[:,0]):.1f} мм")
    print(f"[OK] Диапазон Y (вверх): {np.min(points[:,1]):.1f}..{np.max(points[:,1]):.1f} мм")
    print(f"[OK] Диапазон Z (глубина): {np.min(points[:,2]):.1f}..{np.max(points[:,2]):.1f} мм")
    
    # Центрируем для вращения (сохраняя ориентацию)
    points[:, 0] -= np.mean(points[:, 0])
    points[:, 1] -= np.mean(points[:, 1])
    points[:, 2] -= np.mean(points[:, 2])
    
    # Создаем облако точек Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Визуализация
    print("\n" + "="*60)
    print("3D ОКНО ОТКРЫТО!")
    print("Оси:  X (красная) → вправо, Y (зеленая) → вверх, Z (синяя) → на оператора")
    print(f"Реальный размер: {REAL_LENGTH_MM} мм по вертикали (Y)")
    print("Управление:")
    print("  ЛКМ + двигать = вращение")
    print("  Скролл = зум")
    print("  ПКМ + двигать = перемещение")
    print("  'q' или ESC = выход")
    print("="*60 + "\n")
    
    # Создаем визуализатор
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Скан (X→вправо, Y→вверх, Z→глубина)", width=1400, height=800)
    
    # Добавляем облако
    vis.add_geometry(pcd)
    
    # Настройки рендеринга
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    # Добавляем координатную сетку с ПРАВИЛЬНЫМИ названиями
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Запускаем
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        load_and_view_scan(sys.argv[1])
    else:
        load_and_view_scan("scan_3d.npy")