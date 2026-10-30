"""
Offline test for objects.describe_scene(): fake detections and fake
face results, no camera, no YOLO model needed.

Run: python test_describe_scene.py
"""

import numpy as np

import objects
from config import CALLNAME


def face(name):
    return {
        "name": name,
        "recognized": name != "Unknown",
        "similarity": 0.5,
        "bbox": (0, 0, 10, 10)
    }


def scene(counts, faces=()):
    """Run describe_scene() with faked camera, faces and detections."""

    objects.session = object()
    objects.get_latest_frame = lambda: np.zeros((4, 4, 3), np.uint8)
    objects.get_object_counts = lambda: dict(counts)
    objects.get_face_results = lambda: list(faces)

    return objects.describe_scene()


CASES = [
    ("singular", {"laptop": 1}, (),
     f"I see a laptop, {CALLNAME}."),
    ("an + vowel", {"apple": 1, "oven": 1}, (),
     f"I see an apple and an oven, {CALLNAME}."),
    ("plural", {"cup": 2, "book": 3}, (),
     f"I see two cups and three books, {CALLNAME}."),
    ("mixed list", {"laptop": 1, "cup": 2, "chair": 1}, (),
     f"I see a laptop, two cups and a chair, {CALLNAME}."),
    ("irregular", {"knife": 2, "couch": 2, "mouse": 3}, (),
     f"I see two knives, two couches and three mice, {CALLNAME}."),
    ("multi-word", {"dining table": 2, "cell phone": 4,
                    "hair drier": 2, "toothbrush": 2}, (),
     f"I see two dining tables, four cell phones, two hair driers "
     f"and two toothbrushes, {CALLNAME}."),
    ("scissors", {"scissors": 1}, (),
     f"I see a pair of scissors, {CALLNAME}."),
    ("scissors plural", {"scissors": 2}, (),
     f"I see two pairs of scissors, {CALLNAME}."),
    ("tv", {"tv": 1}, (),
     f"I see a TV, {CALLNAME}."),
    ("ten vs 11", {"book": 10, "cup": 11}, (),
     f"I see ten books and 11 cups, {CALLNAME}."),
    ("empty", {}, (),
     f"I don't see anything I recognize right now, {CALLNAME}."),
    ("people + objects", {"laptop": 1, "cup": 1},
     (face("Unknown"), face("Luka")),
     f"I see Luka, one person I don't recognize, a laptop "
     f"and a cup, {CALLNAME}."),
    ("two unknown", {"chair": 1},
     (face("Unknown"), face("Unknown")),
     f"I see two people I don't recognize and a chair, {CALLNAME}."),
    ("people only", {}, (face("Luka"), face("Ana")),
     f"I see Luka and Ana, {CALLNAME}."),
]


def main():
    failures = 0

    for name, counts, faces, expected in CASES:
        got = scene(counts, faces)
        ok = got == expected
        failures += not ok

        print(f"[{'PASS' if ok else 'FAIL'}] {name:17} {got}")

        if not ok:
            print(f"       expected:         {expected}")

    # Camera not ready.
    objects.get_latest_frame = lambda: None
    got = objects.describe_scene()
    expected = f"I can't see anything, my camera isn't ready, {CALLNAME}."
    failures += got != expected
    print(f"[{'PASS' if got == expected else 'FAIL'}] {'no camera':17} {got}")

    # Detector unavailable.
    objects.session = None
    got = objects.describe_scene()
    expected = f"My object detection isn't available right now, {CALLNAME}."
    failures += got != expected
    print(f"[{'PASS' if got == expected else 'FAIL'}] {'no detector':17} {got}")

    # count_objects: instances per label, most confident first.
    counts = objects.count_objects(
        [("cup", 0.5), ("laptop", 0.9), ("cup", 0.6)]
    )
    ok = list(counts.items()) == [("laptop", 1), ("cup", 2)]
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {'count_objects':17} {counts}")

    print(f"\n{failures} failure(s)")

    return failures


if __name__ == "__main__":
    raise SystemExit(main())
