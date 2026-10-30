import cv2
import numpy as np
import onnxruntime as ort
import glob
import time
import os

# ---------------- CONFIG ----------------

MODEL_PATHS = glob.glob("**/yolo11n.onnx", recursive=True)

if not MODEL_PATHS:
    raise FileNotFoundError(
        "yolo11n.onnx not found in the current project directory."
    )

MODEL_PATH = MODEL_PATHS[0]
CAMERA_ID = 0
INPUT_HEIGHT = 480
INPUT_WIDTH = 640
CONFIDENCE_THRESHOLD = 0.40
PRINT_INTERVAL = 0.5

# COCO class labels used by standard YOLO11n models.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

INDOOR_OBJECTS = {
    # Computers and electronics
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "remote",
    "tv",
    "clock",

    # Furniture
    "chair",
    "couch",
    "bed",
    "dining table",

    # Everyday items
    "bottle",
    "cup",
    "book",
    "scissors",
    "backpack",
    "handbag",
    "suitcase",

    # Kitchen and common areas
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "bowl",
    "fork",
    "knife",
    "spoon",

    # Other indoor objects
    "potted plant",
    "vase",
    "teddy bear",
    "toothbrush",
    "hair drier",
}

# ---------------- MODEL ----------------

print(f"Loading model: {MODEL_PATH}", flush=True)
print("Available providers:", ort.get_available_providers())

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("Session providers:", session.get_providers(), flush=True)

input_info = session.get_inputs()[0]
input_name = input_info.name

print("Model input:", input_info.shape, flush=True)

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    raise RuntimeError(
        f"Cannot open camera {CAMERA_ID}. "
        "Check /dev/video* and try another camera index."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nReal-time detection started.", flush=True)
print("Press Ctrl+C to stop.\n", flush=True)

last_print = 0
frame_count = 0
fps_start = time.perf_counter()

# ---------------- DETECTION LOOP ----------------

try:
    while True:
        ok, frame = camera.read()

        if not ok:
            print("Camera frame unavailable.", flush=True)
            time.sleep(0.1)
            continue

        # Resize and normalize image.
        image = cv2.resize(
            frame, (INPUT_WIDTH, INPUT_HEIGHT)
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)

        # GPU inference when supported.
        outputs = session.run(
            None, {input_name: image}
        )

        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)

        detections = []

        # Handle standard YOLO output: [84, 8400]
        # or [8400, 84].
        if pred.ndim == 2 and (
            pred.shape[0] == len(COCO_CLASSES) + 4
            or pred.shape[1] == len(COCO_CLASSES) + 4
        ):
            if pred.shape[0] == len(COCO_CLASSES) + 4:
                pred = pred.T

            boxes = pred[:, :4]
            scores = pred[:, 4:]

            class_ids = np.argmax(scores, axis=1)
            confidences = scores[
                np.arange(len(scores)), class_ids
            ]

            valid = confidences >= CONFIDENCE_THRESHOLD
            boxes = boxes[valid]
            confidences = confidences[valid]
            class_ids = class_ids[valid]

            # YOLO center-x, center-y, width, height
            # -> x, y, width, height for NMS.
            xywh = np.empty_like(boxes)
            xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            xywh[:, 2] = boxes[:, 2]
            xywh[:, 3] = boxes[:, 3]

            indices = cv2.dnn.NMSBoxes(
                xywh.tolist(),
                confidences.tolist(),
                CONFIDENCE_THRESHOLD,
                0.45
            )

            if len(indices) > 0:
                for idx in np.asarray(indices).reshape(-1):
                    class_id = int(class_ids[idx])

                    if class_id < len(COCO_CLASSES):
                        label = COCO_CLASSES[class_id]

                        if label in INDOOR_OBJECTS:
                            detections.append((label, float(confidences[idx])))

        # Handle models exported with NMS:
        # rows formatted as x1, y1, x2, y2, confidence, class_id.
        elif pred.ndim == 2 and pred.shape[1] == 6:
            for row in pred:
                confidence = float(row[4])
                class_id = int(row[5])

                if (
                    confidence >= CONFIDENCE_THRESHOLD
                    and 0 <= class_id < len(COCO_CLASSES)
                ):
                    detections.append((
                        COCO_CLASSES[class_id],
                        confidence
                    ))

        frame_count += 1
        now = time.perf_counter()

        if now - last_print >= PRINT_INTERVAL:
            fps = frame_count / max(now - fps_start, 1e-6)

            if detections:
                # Keep the highest-confidence result per class.
                best = {}

                for label, confidence in detections:
                    if (
                        label not in best
                        or confidence > best[label]
                    ):
                        best[label] = confidence

                description = ", ".join(
                    f"{label} ({confidence:.0%})"
                    for label, confidence in sorted(
                        best.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                )
            else:
                description = "no recognized objects"

            timestamp = time.strftime("%H:%M:%S")

            print(
                f"[{timestamp}] FPS: {fps:.1f} | "
                f"I see: {description}",
                flush=True
            )

            last_print = now
            frame_count = 0
            fps_start = now

except KeyboardInterrupt:
    print("\nDetection stopped.", flush=True)

finally:
    camera.release()