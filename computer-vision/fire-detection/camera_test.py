import cv2


# 0 usually means the laptop's first available camera.
CAMERA_INDEX = 0

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: The camera could not be opened.")
    print("Try changing CAMERA_INDEX to 1 or 2.")
    raise SystemExit

print("Camera opened successfully.")
print("Press Q while looking at the camera window to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: A camera frame could not be read.")
        break

    cv2.imshow("Fire Tower - Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()