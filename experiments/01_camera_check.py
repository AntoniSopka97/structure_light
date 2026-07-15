import cv2

# Меняем индексы на правильные видео-ноды
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(1, cv2.CAP_V4L2)

# Ставим MJPEG, чтобы пролез Full HD
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print(f"Камера Левая (4) статус: {cap1.isOpened()}")
print(f"Камера Правая (2) статус: {cap2.isOpened()}")

if cap1.isOpened() and cap2.isOpened():
    print("Нащупали верные usb. Запускаем превью...")
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if ret1 and ret2:
            cv2.imshow("LEFT", cv2.resize(f1, (640, 360)))
            cv2.imshow("RIGHT", cv2.resize(f2, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
