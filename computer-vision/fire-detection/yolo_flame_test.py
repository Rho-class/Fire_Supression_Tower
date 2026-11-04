import time

import cv2
from ultralytics import YOLOWorld


CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.20

# YOLO-World can search for objects using text descriptions.
model = YOLOWorld("yolov8s-world.pt")

# Classes that matter to the fire-suppression tower.
model.set_classes(["fire", "flame", "person"])

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("YOLO flame detector started.")
print("The model may download during the first run.")
print("Press Q to close the program.")

previous_time = time.perf_counter()

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Camera frame could not be read.")
        break

    # Give the current camera frame to YOLO.
    result = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False,
    )[0]

    fire_detected = False
    person_detected = False
    strongest_fire_confidence = 0.0
    fire_center = None

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if class_name in ("fire", "flame"):
                fire_detected = True
                color = (0, 165, 255)

                if confidence > strongest_fire_confidence:
                    strongest_fire_confidence = confidence
                    fire_center = (center_x, center_y)

            elif class_name == "person":
                person_detected = True
                color = (0, 0, 255)

            else:
                color = (255, 255, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                color,
                -1,
            )

    # Person detection takes priority as a no-spray lockout.
    if person_detected:
        status = "PERSON DETECTED - NO SPRAY"
        status_color = (0, 0, 255)

    elif fire_detected:
        status = "FIRE CANDIDATE DETECTED"
        status_color = (0, 165, 255)

    else:
        status = "SCANNING"
        status_color = (0, 255, 0)

    current_time = time.perf_counter()
    fps = 1.0 / max(current_time - previous_time, 0.0001)
    previous_time = current_time

    cv2.rectangle(frame, (0, 0), (640, 55), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (535, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    if fire_center is not None:
        print(
            {
                "detected": True,
                "confidence": round(strongest_fire_confidence, 3),
                "center_pixel": fire_center,
                "person_lockout": person_detected,
            }
        )

    cv2.imshow("Fire Tower - YOLO Flame Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()