import cv2
import json
import os
import threading
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTOR_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_recognition_sface_2021dec.onnx"
)

DATABASE_FILE = os.path.join(BASE_DIR, "faces.json")

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# OpenCV cosine similarity threshold.
FACE_THRESHOLD = 0.363

face_count = 0

# Backward-compatible list of names.
# Example: ["Luka", "Unknown"]
recognized_people = []

# Detailed results for every detected face.
face_results = []

known_faces = {}

last_face_features = []

# Protect data accessed by the camera and voice threads.
state_lock = threading.Lock()

# Signals the camera thread to stop.
camera_stop_event = threading.Event()


def stop_face_detection():
    """Request a clean shutdown of the camera thread."""
    camera_stop_event.set()
    print("[FRIDAY] Camera shutdown requested.")

# Registration synchronization.
registration_lock = threading.Lock()
registration_event = threading.Event()

registration_request = None
registration_result = None

def create_models():
    """Load the YuNet detector and SFace recognizer."""

    for model_path in (DETECTOR_MODEL, RECOGNIZER_MODEL):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Missing face recognition model: {model_path}"
            )

    detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        RECOGNIZER_MODEL,
        ""
    )

    print("[FRIDAY] YuNet face detector loaded.")
    print("[FRIDAY] SFace recognizer loaded.")

    return detector, recognizer

def load_faces():
    """Load registered face encodings from faces.json."""

    global known_faces

    if not os.path.exists(DATABASE_FILE):
        known_faces = {}
        print("[FRIDAY] No saved faces found.")
        return

    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Face database must contain a JSON object."
            )

        known_faces = data

        print(
            f"[FRIDAY] Loaded {len(known_faces)} registered people."
        )

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"[FRIDAY] Database error: {error}")
        known_faces = {}


def save_faces():
    """Save face encodings safely to disk."""

    temporary_file = DATABASE_FILE + ".tmp"

    try:
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(known_faces, file, indent=2)

        os.replace(temporary_file, DATABASE_FILE)

        print("[FRIDAY] Face database saved.")
        return True

    except (OSError, TypeError, ValueError) as error:
        print(f"[FRIDAY] Could not save database: {error}")

        if os.path.exists(temporary_file):
            try:
                os.remove(temporary_file)
            except OSError:
                pass

        return False

def identify_face(feature, recognizer):
    """
    Compare one detected face against all registered faces.

    Returns:
        (name, similarity)

    Unknown faces return ("Unknown", best_similarity).
    """

    best_name = "Unknown"
    best_similarity = -1.0

    for name, saved_features in known_faces.items():

        # Support both a list of encodings and a single encoding.
        if (
            isinstance(saved_features, list)
            and saved_features
            and isinstance(saved_features[0], (int, float))
        ):
            saved_features = [saved_features]

        for saved_feature in saved_features:

            try:
                saved_feature = np.asarray(
                    saved_feature,
                    dtype=np.float32
                ).reshape(1, -1)

                similarity = float(
                    recognizer.match(
                        feature,
                        saved_feature,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name

            except (ValueError, TypeError, cv2.error) as error:
                print(
                    f"[FRIDAY] Could not compare saved face "
                    f"for {name}: {error}"
                )

    if best_similarity >= FACE_THRESHOLD:
        return best_name, best_similarity

    return "Unknown", best_similarity

def register_person_by_name(name, timeout=10):
    global registration_request
    global registration_result

    name = name.strip()

    if not name:
        return False, "No name was provided."

    with registration_lock:

        if registration_request is not None:
            return (
                False,
                "Another registration is already in progress."
            )

        registration_request = name
        registration_result = None
        registration_event.clear()

    print(f"[FRIDAY] Waiting to register a face for {name}...")

    if not registration_event.wait(timeout):

        with registration_lock:
            if registration_request == name:
                registration_request = None

        return (
            False,
            "Registration timed out. Make sure exactly one face "
            "is visible and try again."
        )

    with registration_lock:
        result = registration_result

    if result is None:
        return False, "Face registration failed."

    return result


def process_registration(current_features):
    global registration_request
    global registration_result

    with registration_lock:
        requested_name = registration_request

    if requested_name is None:
        return

    # Register only when exactly one face is visible.
    if len(current_features) != 1:
        return

    feature = current_features[0]

    # Save the face encoding.
    known_faces.setdefault(requested_name, []).append(
        feature.flatten().astype(float).tolist()
    )

    if not save_faces():

        # Undo the addition if the database could not be saved.
        known_faces[requested_name].pop()

        if not known_faces[requested_name]:
            del known_faces[requested_name]

        with registration_lock:
            if registration_request == requested_name:
                registration_request = None
                registration_result = (
                    False,
                    "Could not save the face database."
                )
                registration_event.set()

        return

    with registration_lock:

        if registration_request == requested_name:
            registration_request = None
            registration_result = (True, requested_name)
            registration_event.set()

    print(f"[FRIDAY] Registered {requested_name}.")

def get_face_count():
    with state_lock:
        return face_count


def get_recognized_people():
    with state_lock:
        return recognized_people.copy()


def get_face_results():

    with state_lock:
        return [
            result.copy()
            for result in face_results
        ]

def start_face_detection():
    global face_count
    global recognized_people
    global face_results
    global last_face_features

    detector, recognizer = create_models()

    load_faces()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[FRIDAY] Error: Could not open camera.")
        cap.release()
        return

    print("\n========== FRIDAY VISION ==========")
    print("Headless mode enabled. No GUI will be created.")
    print("Multiple-face recognition enabled.")
    print("Voice registration is enabled.")
    print("===================================\n")

    camera_stop_event.clear()

    try:
        while not camera_stop_event.is_set():

            ret, frame = cap.read()

            # Stop promptly if shutdown was requested during capture.
            if camera_stop_event.is_set():
                break

            if not ret:
                print(
                    "\n[FRIDAY] Warning: Could not read camera frame."
                )
                time.sleep(0.1)
                continue

            # Mirror the camera image.
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # YuNet needs the current image dimensions.
            detector.setInputSize((width, height))

            # Detect faces.
            _, faces = detector.detect(frame)

            current_results = []
            current_features = []

            # Process every face independently.
            if faces is not None:

                for face in faces:

                    try:
                        aligned_face = recognizer.alignCrop(
                            frame,
                            face
                        )

                        feature = recognizer.feature(aligned_face)

                        name, similarity = identify_face(
                            feature,
                            recognizer
                        )

                    except cv2.error as error:
                        print(
                            f"[FRIDAY] Face processing error: {error}"
                        )
                        continue

                    x, y, w, h = face[:4].astype(int)

                    result = {
                        "name": name,
                        "recognized": name != "Unknown",
                        "similarity": float(similarity),
                        "bbox": (
                            int(x),
                            int(y),
                            int(w),
                            int(h)
                        )
                    }

                    current_results.append(result)
                    current_features.append(feature.copy())

            # Publish a complete snapshot of the current frame.
            with state_lock:
                face_count = len(current_results)
                face_results = current_results.copy()

                recognized_people = [
                    result["name"]
                    for result in current_results
                ]

                last_face_features = current_features.copy()

            # Process registration using this frame's features.
            process_registration(current_features)

            # Print status only when face information changes.
            known_count = sum(
                1
                for result in current_results
                if result["recognized"]
            )

            unknown_count = len(current_results) - known_count

            print(
                f"\r[FRIDAY Vision] Faces: {len(current_results)} | "
                f"Known: {known_count} | Unknown: {unknown_count}",
                end="",
                flush=True
            )

    except Exception as error:
        print(f"\n[FRIDAY] Camera error: {error}")

    finally:
        camera_stop_event.set()

        cap.release()

        with state_lock:
            face_count = 0
            recognized_people = []
            face_results = []
            last_face_features = []

        print("\n[FRIDAY] Vision system stopped.")

if __name__ == "__main__":
    start_face_detection()