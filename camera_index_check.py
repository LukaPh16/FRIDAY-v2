import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera index {i}: FOUND")
        ret, frame = cap.read()
        print(f"  Frame available: {ret}")
    else:
        print(f"Camera index {i}: not available")

    cap.release()
