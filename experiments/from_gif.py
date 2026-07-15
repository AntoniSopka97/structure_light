import numpy as np
import open3d as o3d
import imageio
import os
from PIL import Image, ImageDraw, ImageFont

def create_gif_auto_rotate(npy_file="scan_3d.npy", output_gif="3d_scan_auto.gif", n_frames=72):
    """
    Создает GIF с автоматическим вращением через Open3D
    """
    
    print("[3D] Загрузка файла...")
    
    try:
        profiles = np.load(npy_file)
        print(f"[ЗАГРУЗКА] {npy_file} shape: {profiles.shape}")
    except:
        print("[ОШИБКА] Файл не найден!")
        return
    
    n_frames_data, n_points = profiles.shape
    
    # --- РЕАЛЬНЫЕ РАЗМЕРЫ ---
    REAL_LENGTH_MM = 400.0
    REAL_WIDTH_MM = 100.0
    HEIGHT_SCALE = 3.0
    
    z_min, z_max = np.min(profiles), np.max(profiles)
    real_height_mm = (z_max - z_min) * HEIGHT_SCALE
    
    print(f"[OK] Размеры: Длина={REAL_LENGTH_MM}мм, Ширина={REAL_WIDTH_MM}мм, Высота={real_height_mm:.1f}мм")
    
    # --- ПРАВИЛЬНАЯ ОРИЕНТАЦИЯ ---
    # X - ширина, Y - длина скана (была вертикальная), Z - рельеф
    # Но чтобы сцена лежала - меняем Y и Z местами
    
    x_coords = np.linspace(-REAL_WIDTH_MM/2, REAL_WIDTH_MM/2, n_points)
    y_coords = np.linspace(0, REAL_LENGTH_MM, n_frames_data)
    
    points = []
    colors = []
    z_range = z_max - z_min if z_max > z_min else 1.0
    
    for i in range(n_frames_data):
        for j in range(n_points):
            x = x_coords[j]
            y = y_coords[i]
            z = profiles[i, j] * HEIGHT_SCALE
            
            if abs(z) < 100:
                # Оставляем как есть: X→ширина, Y→длина, Z→рельеф
                points.append([x, y, z])  # <-- НЕ МЕНЯЕМ!
                
                norm_z = (z - z_min * HEIGHT_SCALE) / (z_range * HEIGHT_SCALE)
                norm_z = np.clip(norm_z, 0, 1)
                
                if norm_z < 0.5:
                    r = 0
                    g = int(255 * norm_z / 0.5)
                    b = 255
                else:
                    r = int(255 * (norm_z - 0.5) / 0.5)
                    g = int(255 * (1 - (norm_z - 0.5) / 0.5))
                    b = 0
                colors.append([r/255, g/255, b/255])
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Центрируем
    points[:, 0] -= np.mean(points[:, 0])
    points[:, 1] -= np.mean(points[:, 1])
    points[:, 2] -= np.mean(points[:, 2])
    
    print(f"[OK] Создано {len(points)} точек")
    print(f"[OK] X (ширина): {np.min(points[:,0]):.1f}..{np.max(points[:,0]):.1f} мм")
    print(f"[OK] Y (длина): {np.min(points[:,1]):.1f}..{np.max(points[:,1]):.1f} мм")
    print(f"[OK] Z (рельеф): {np.min(points[:,2]):.1f}..{np.max(points[:,2]):.1f} мм")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # --- СОЗДАЕМ ВИЗУАЛИЗАТОР С АВТО-ВРАЩЕНИЕМ ---
    print(f"[ГИФКА] Рендеринг {n_frames} кадров...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.add_geometry(pcd)
    
    # Добавляем координатную сетку
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    vis.add_geometry(coord_frame)
    
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    # --- ИСПОЛЬЗУЕМ ВСТРОЕННОЕ ВРАЩЕНИЕ ---
    ctrl = vis.get_view_control()
    
    # Ставим камеру на хорошую позицию
    ctrl.set_lookat([0, 0, 0])
    ctrl.set_up([0, 1, 0])
    ctrl.set_front([0, 0, -1])
    
    frames = []
    
    for i in range(n_frames):
        # Просто вращаем через встроенную функцию
        angle = i * (360.0 / n_frames)
        
        # Ставим камеру на окружность
        radius = 500
        cam_x = radius * np.sin(np.radians(angle))
        cam_z = radius * np.cos(np.radians(angle))
        
        # Направляем камеру из точки на центр
        front = np.array([-cam_x, 0, -cam_z])
        front = front / np.linalg.norm(front)
        
        # Устанавливаем
        ctrl.set_front(front)
        ctrl.set_lookat([0, 0, 0])
        ctrl.set_up([0, 1, 0])
        
        vis.poll_events()
        vis.update_renderer()
        
        img = vis.capture_screen_float_buffer(do_render=True)
        if img is not None:
            img = (np.asarray(img) * 255).astype(np.uint8)
            frames.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"  Кадр {i+1}/{n_frames}")
    
    vis.destroy_window()
    
    if len(frames) == 0:
        print("[ОШИБКА] Нет кадров!")
        return
    
    # --- НАКЛАДЫВАЕМ ТЕКСТ ---
    print("[ТЕКСТ] Добавляю размеры...")
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    info_lines = [
        f"Длина: {REAL_LENGTH_MM:.0f} мм",
        f"Ширина: {REAL_WIDTH_MM:.0f} мм", 
        f"Высота: {real_height_mm:.1f} мм",
        f"Точек: {len(points)}"
    ]
    
    frames_with_text = []
    
    for frame in frames:
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        
        draw.rectangle([(10, 10), (230, 120)], fill=(0, 0, 0, 200))
        
        y_offset = 18
        for line in info_lines:
            draw.text((18, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 24
        
        draw.rectangle([(10, 10), (230, 120)], outline=(0, 180, 255), width=2)
        
        frames_with_text.append(np.array(pil_img))
    
    print(f"[СОХРАНЕНИЕ] {output_gif}...")
    imageio.mimsave(output_gif, frames_with_text, fps=12, loop=0)
    
    print(f"[УСПЕХ] Гифка создана: {output_gif}")
    print(f"[ИНФО] Размер: {os.path.getsize(output_gif) / 1024:.1f} KB")
    print("[ИНФО] Используй координатную сетку для ориентира")

if __name__ == "__main__":
    create_gif_auto_rotate("scan_3d.npy", "3d_scan_auto.gif", n_frames=72)