import typer
import cv2
import supervision as sv
from ultralytics import YOLO
import os

# Проверяем существование модели
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights file '{MODEL_PATH}' not found.")

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)
app = typer.Typer()

def process_video(input_video: str, output_file: str):
    if not os.path.exists(input_video):
        print(f"Error: Video file '{input_video}' not found.")
        return

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_video}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Если fps неопределён, берём 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра моделью YOLO
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Аннотирование кадра
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        out.write(annotated_frame)

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_file}")

@app.command()
def process(input_video: str, output_file: str = "output.mp4"):
    typer.echo(f"Processing video: {input_video}")
    process_video(input_video, output_file)

if __name__ == "__main__":
    app()
