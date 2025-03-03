import typer
import cv2
import supervision as sv
from ultralytics import YOLO
import os

# Ограничения на размер видео
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

# Проверяем существование модели
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights file '{MODEL_PATH}' not found.")

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)
app = typer.Typer()


def check_and_resize_video(input_video: str, temp_video: str) -> str:
    """Проверяет размер видео и изменяет его при необходимости."""
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_video}'.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if width == TARGET_WIDTH and height == TARGET_HEIGHT:
        cap.release()
        return input_video  # Размер уже соответствует, обработаем оригинал

    print(f"Resizing video from {width}x{height} to {TARGET_WIDTH}x{TARGET_HEIGHT}...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        out.write(resized_frame)

    cap.release()
    out.release()

    return temp_video


def process_video(input_video: str, output_file: str):
    temp_video = "temp_resized.mp4"
    video_to_process = check_and_resize_video(input_video, temp_video)

    if not video_to_process:
        return

    cap = cv2.VideoCapture(video_to_process)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка YOLO
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Аннотирование
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        out.write(annotated_frame)

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Удаляем временный файл, если изменяли размер
    if video_to_process == temp_video:
        os.remove(temp_video)

    print(f"Processing complete. Output saved to {output_file}")


@app.command()
def process(input_video: str, output_file: str = "output.mp4"):
    typer.echo(f"Processing video: {input_video}")
    process_video(input_video, output_file)


if __name__ == "__main__":
    app()
