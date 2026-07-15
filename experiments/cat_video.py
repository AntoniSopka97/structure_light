import cv2
import os
import sys

def trim_video(input_file="Screencast from 2026-07-15 23-28-09.webm", 
               output_file="scan_trimmed.mp4",
               start_trim=5.0,    # Удалить первые 5 секунд
               end_trim=3.0):     # Удалить последние 3 секунды
    
    print(f"[ВИДЕО] Обрезаем: {input_file}")
    print(f"[ВИДЕО] Удаляем первые {start_trim}с и последние {end_trim}с")
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        print(f"[ОШИБКА] Файл {input_file} не найден!")
        return
    
    # Открываем видео
    cap = cv2.VideoCapture(input_file)
    
    if not cap.isOpened():
        print("[ОШИБКА] Не удалось открыть видео!")
        return
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps
    
    print(f"[ИНФО] FPS: {fps:.2f}")
    print(f"[ИНФО] Всего кадров: {total_frames}")
    print(f"[ИНФО] Длительность: {total_duration:.2f}с")
    print(f"[ИНФО] Размер: {width}x{height}")
    
    # Вычисляем кадры для обрезки
    start_frame = int(start_trim * fps)
    end_frame = int((total_duration - end_trim) * fps)
    new_total_frames = end_frame - start_frame
    new_duration = new_total_frames / fps
    
    print(f"[ИНФО] Начинаем с кадра: {start_frame}")
    print(f"[ИНФО] Заканчиваем кадром: {end_frame}")
    print(f"[ИНФО] Новая длительность: {new_duration:.2f}с")
    
    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Пропускаем первые кадры
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_written = 0
    current_frame = start_frame
    
    print("[ОБРЕЗКА] Обрабатываю кадры...")
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frames_written += 1
        current_frame += 1
        
        # Прогресс
        if frames_written % 100 == 0:
            progress = (frames_written / new_total_frames) * 100
            print(f"  Прогресс: {progress:.1f}% ({frames_written}/{new_total_frames})")
    
    cap.release()
    out.release()
    
    print(f"[УСПЕХ] Видео обрезано!")
    print(f"[УСПЕХ] Сохранено: {output_file}")
    print(f"[ИНФО] Кадров: {frames_written}")
    print(f"[ИНФО] Длительность: {frames_written / fps:.2f}с")
    
    # Показываем информацию о новом файле
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[ИНФО] Размер: {size_mb:.1f} MB")

if __name__ == "__main__":
    # Твой файл
    input_video = "Screencast from 2026-07-15 23-28-09.webm"
    
    # Обрезаем
    trim_video(
        input_file=input_video,
        output_file="scan_trimmed.mp4",
        start_trim=5.0,  # Убираем первые 5 секунд
        end_trim=3.0     # Убираем последние 3 секунды
    )
    
    # Если хочешь сохранить в WebM (как исходник)
    # trim_video(
    #     input_file=input_video,
    #     output_file="scan_trimmed.webm",
    #     start_trim=5.0,
    #     end_trim=3.0
    # )