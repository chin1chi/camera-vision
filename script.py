import cv2
import torch
from ultralytics import YOLO

# Загрузка модели
model_path = "best.pt"  # Путь к модели
model = YOLO(model_path)

# Открытие видеофайла
video_path = "slp.jpeg"  # Укажи путь к своему видео
cap = cv2.VideoCapture(video_path)

# Получение параметров видео
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Настройка записи выходного видео
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourсc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предсказание
    results = model(frame, conf=0.3)

    # Отрисовка результатов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            if confidence > 0.3:  # Фильтр по порогу уверенности
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Sleep Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработанное видео сохранено в {output_path}")

