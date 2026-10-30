import os
import threading
import time

import cv2
import numpy as np

from camera import get_face_results, get_latest_frame
from config import CALLNAME, OBJECT_DETECTION_FPS

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL = os.path.join(BASE_DIR, "models", "yolo11n.onnx")

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

# Used when the model has dynamic input dimensions.
DEFAULT_INPUT_SIZE = (640, 640)

# Stored results older than this are not trusted.
MAX_RESULT_AGE = 2.0

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

# "person" is left out on purpose: people come from face
# recognition.
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

# Labels that read better spoken a different way.
SPOKEN_NAMES = {
    "tv": "TV",
}

IRREGULAR_PLURALS = {
    "knife": "knives",
    "mouse": "mice",
}

NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten"
]

# Set by load_object_detector(). None means detection is disabled.
session = None
input_name = None
input_width, input_height = DEFAULT_INPUT_SIZE

# Latest background detection: {label: count}, ordered by
# confidence, and the time.monotonic() it was stored.
object_lock = threading.Lock()
object_counts = None
object_counts_time = 0.0

# Signals the object detection thread to stop.
object_stop_event = threading.Event()


def load_object_detector():
    """
    Load YOLO and warm it up once. Returns True on success.

    On failure detection stays disabled and FRIDAY keeps working.
    """

    global session
    global input_name
    global input_width
    global input_height

    print("[FRIDAY] Loading YOLO...")

    if ort is None:
        print(
            "[FRIDAY] Warning: onnxruntime is not installed. "
            "Object detection disabled."
        )
        return False

    if not os.path.isfile(YOLO_MODEL):
        print(
            f"[FRIDAY] Warning: Missing YOLO model: {YOLO_MODEL}. "
            "Object detection disabled."
        )
        return False

    try:
        new_session = ort.InferenceSession(
            YOLO_MODEL,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

        input_info = new_session.get_inputs()[0]

        # Expected shape: [batch, channels, height, width].
        height, width = input_info.shape[2:4]

        if isinstance(height, int) and isinstance(width, int):
            input_width, input_height = width, height
        else:
            input_width, input_height = DEFAULT_INPUT_SIZE

        input_name = input_info.name

        # Warm up so the first real question isn't slow.
        blank = np.zeros((input_height, input_width, 3), np.uint8)

        new_session.run(None, {input_name: preprocess(blank)})

    except Exception as error:
        print(
            f"[FRIDAY] Warning: Could not load YOLO: {error}. "
            "Object detection disabled."
        )
        return False

    session = new_session

    print(
        f"[FRIDAY] YOLO ready ({input_width}x{input_height}, "
        f"{session.get_providers()[0]})."
    )

    return True


def is_object_detection_available():
    return session is not None


def preprocess(frame):
    """Resize and normalize a BGR frame into a YOLO input tensor."""

    image = cv2.resize(frame, (input_width, input_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)


def detect_objects(frame):
    """
    Run YOLO on one BGR frame.

    Returns:
        list of (label, confidence) for indoor objects, one entry
        per instance after NMS.
    """

    if session is None:
        return []

    outputs = session.run(None, {input_name: preprocess(frame)})

    pred = np.squeeze(np.asarray(outputs[0]))

    # A model exported with NMS that found one object squeezes
    # down to a single row.
    if pred.ndim == 1 and pred.shape[0] == 6:
        pred = pred.reshape(1, 6)

    detections = []

    # Handle standard YOLO output: [84, N] or [N, 84].
    if pred.ndim == 2 and (
        pred.shape[0] == len(COCO_CLASSES) + 4
        or pred.shape[1] == len(COCO_CLASSES) + 4
    ):
        if pred.shape[0] == len(COCO_CLASSES) + 4:
            pred = pred.T

        boxes = pred[:, :4]
        scores = pred[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

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
            NMS_THRESHOLD
        )

        for idx in np.asarray(indices).reshape(-1):
            class_id = int(class_ids[idx])

            if class_id < len(COCO_CLASSES):
                label = COCO_CLASSES[class_id]

                if label in INDOOR_OBJECTS:
                    detections.append(
                        (label, float(confidences[idx]))
                    )

    # Handle models exported with NMS:
    # rows formatted as x1, y1, x2, y2, confidence, class_id.
    elif pred.ndim == 2 and pred.shape[1] == 6:
        for row in pred:
            confidence = float(row[4])
            class_id = int(row[5])

            if (
                confidence >= CONFIDENCE_THRESHOLD
                and 0 <= class_id < len(COCO_CLASSES)
                and COCO_CLASSES[class_id] in INDOOR_OBJECTS
            ):
                detections.append(
                    (COCO_CLASSES[class_id], confidence)
                )

    return detections


def count_objects(detections):
    """
    Count instances per label, most confident label first.

    Returns:
        {label: count}
    """

    best = {}
    counts = {}

    for label, confidence in detections:
        counts[label] = counts.get(label, 0) + 1
        best[label] = max(confidence, best.get(label, 0.0))

    return {
        label: counts[label]
        for label in sorted(best, key=best.get, reverse=True)
    }


def start_object_detection():
    """
    Background loop: detect objects in the latest camera frame
    about OBJECT_DETECTION_FPS times per second.
    """

    global object_counts
    global object_counts_time

    if session is None:
        return

    interval = 1.0 / max(OBJECT_DETECTION_FPS, 0.1)
    last_error = None

    # Don't clear object_stop_event here: a stop requested before
    # the thread started must not be lost.

    try:
        while not object_stop_event.is_set():

            started = time.monotonic()

            frame = get_latest_frame()

            if frame is not None:
                try:
                    counts = count_objects(detect_objects(frame))

                    with object_lock:
                        object_counts = counts
                        object_counts_time = time.monotonic()

                    last_error = None

                except Exception as error:

                    # Report each new error once, not every frame.
                    if str(error) != last_error:
                        print(
                            f"\n[FRIDAY] Object detection error: "
                            f"{error}"
                        )
                        last_error = str(error)

            elapsed = time.monotonic() - started

            object_stop_event.wait(max(0.0, interval - elapsed))

    finally:
        with object_lock:
            object_counts = None
            object_counts_time = 0.0


def stop_object_detection():
    """Request a clean shutdown of the object detection thread."""
    object_stop_event.set()
    print("[FRIDAY] Object detection shutdown requested.")


def get_object_counts():
    """
    Return the latest {label: count}, or None if the background
    results are missing or older than MAX_RESULT_AGE.
    """

    with object_lock:
        if object_counts is None:
            return None

        if time.monotonic() - object_counts_time > MAX_RESULT_AGE:
            return None

        return dict(object_counts)


def count_word(count):
    if count < len(NUMBER_WORDS):
        return NUMBER_WORDS[count]

    return str(count)


def pluralize(noun):
    """Plural of a single word or the last word of a phrase."""

    head, _, last = noun.rpartition(" ")

    if last in IRREGULAR_PLURALS:
        last = IRREGULAR_PLURALS[last]
    elif last.endswith(("s", "sh", "ch", "x", "z")):
        last += "es"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"

    return f"{head} {last}" if head else last


def describe_object(label, count):
    """'a laptop', 'an apple', 'two cups', 'a pair of scissors'."""

    noun = SPOKEN_NAMES.get(label, label)

    if label == "scissors":
        if count == 1:
            return "a pair of scissors"

        return f"{count_word(count)} pairs of scissors"

    if count == 1:
        article = "an" if noun[0].lower() in "aeiou" else "a"
        return f"{article} {noun}"

    return f"{count_word(count)} {pluralize(noun)}"


def join_phrases(phrases):
    """'a', 'a and b', 'a, b and c'."""

    if len(phrases) == 1:
        return phrases[0]

    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def build_scene_sentence(counts, faces):
    """
    Build the spoken answer from object counts ({label: count})
    and face recognition results (get_face_results() format).
    """

    phrases = []

    # Named people first, each name once, in the order seen.
    for result in faces:
        if result["recognized"] and result["name"] not in phrases:
            phrases.append(result["name"])

    unknown_count = sum(
        1
        for result in faces
        if not result["recognized"]
    )

    if unknown_count == 1:
        phrases.append("one person I don't recognize")

    elif unknown_count > 1:
        phrases.append(
            f"{count_word(unknown_count)} people I don't recognize"
        )

    for label, count in counts.items():
        if count > 0:
            phrases.append(describe_object(label, count))

    if not phrases:
        return f"I don't see anything I recognize right now, {CALLNAME}."

    return f"I see {join_phrases(phrases)}, {CALLNAME}."


def describe_scene():
    """Answer "what do you see" without going through the LLM."""

    if not is_object_detection_available():
        return f"My object detection isn't available right now, {CALLNAME}."

    frame = get_latest_frame()

    if frame is None:
        return f"I can't see anything, my camera isn't ready, {CALLNAME}."

    counts = get_object_counts()

    # Background results are stale or missing: detect once now.
    if counts is None:
        try:
            counts = count_objects(detect_objects(frame))

        except Exception as error:
            print(f"[FRIDAY] Object detection error: {error}")
            return (
                f"My object detection isn't available right now, "
                f"{CALLNAME}."
            )

    return build_scene_sentence(counts, get_face_results())
