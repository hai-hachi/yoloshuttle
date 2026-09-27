from ultralytics import YOLO
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python test_shuttle.py MODEL.pt")
    sys.exit(1)

MODEL_PATH = sys.argv[1]

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.10,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    max_conf = 0.0

    if result.boxes is not None and len(result.boxes) > 0:
        max_conf = max(float(x) for x in result.boxes.conf)

    cv2.putText(
        annotated,
        f"Max conf: {max_conf:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(MODEL_PATH, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()