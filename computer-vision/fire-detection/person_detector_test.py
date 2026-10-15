import time

import cv2


CAMERA_INDEX = 0

# Open the laptop camera.
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

# Create OpenCV's built-in person detector.
person_detector = cv2.HOGDescriptor()
person_detector.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

previous_time = time.perf_counter()

print("Person detector started.")
print("Stand far enough away for the camera to see most of your body.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Camera frame could not be read.")
        break

    # Smaller frames run faster.
    frame = cv2.resize(frame, (640, 480))

    # Search the current frame for people.
    boxes, scores = person_detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    # Remove overlapping duplicate boxes.
    boxes_list = list(boxes)
    scores_list = [float(score) for score in scores]

    selected_boxes = cv2.dnn.NMSBoxes(
        boxes_list,
        scores_list,
        score_threshold=0.0,
        nms_threshold=0.4,
    )

    person_count = 0

    for index in selected_boxes:
        index = int(index)
        x, y, width, height = boxes_list[index]

        person_count += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            frame,
            "PERSON",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if person_count > 0:
        status = f"NO SPRAY: {person_count} PERSON(S) DETECTED"
        status_color = (0, 0, 255)
    else:
        status = "NO PERSON DETECTED"
        status_color = (0, 255, 0)

    current_time = time.perf_counter()
    elapsed_time = current_time - previous_time
    previous_time = current_time

    fps = 1.0 / max(elapsed_time, 0.0001)

    cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)

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
        (530, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Fire Tower - Person Detector Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
