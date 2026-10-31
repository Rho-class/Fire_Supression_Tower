"""Review captured flame photos and save selected non-flame regions as negative crops."""
from pathlib import Path
from datetime import datetime
import cv2

HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets" / "candle-fire" / "train"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
WINDOW = "Hard-negative crop reviewer"
MAX_W, MAX_H = 1180, 760
files = sorted(IMAGES.glob("webcam_*.jpg"))
if not files:
    raise SystemExit(f"No webcam captures found in {IMAGES}")

index = 0
frame = None
scale = 1.0
boxes = []
start = None
preview = None


def mouse(event, x, y, _flags, _param):
    global start, preview
    if frame is None:
        return
    fx, fy = min(int(x / scale), frame.shape[1] - 1), min(int(y / scale), frame.shape[0] - 1)
    if event == cv2.EVENT_LBUTTONDOWN:
        start = (fx, fy)
        preview = (fx, fy, fx, fy)
    elif event == cv2.EVENT_MOUSEMOVE and start is not None:
        preview = (start[0], start[1], fx, fy)
    elif event == cv2.EVENT_LBUTTONUP and start is not None:
        x1, x2 = sorted((start[0], fx))
        y1, y2 = sorted((start[1], fy))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))
        start = None
        preview = None


def load_current():
    global frame, scale, boxes, start, preview
    frame = cv2.imread(str(files[index]))
    if frame is None:
        raise RuntimeError(f"Could not read {files[index]}")
    h, w = frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    boxes, start, preview = [], None, None


def display():
    view = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(view, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (0, 255, 255), 2)
        cv2.putText(view, str(i), (int(x1*scale), max(20, int(y1*scale)-5)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    if preview:
        x1,y1,x2,y2 = preview
        cv2.rectangle(view, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (0,255,0), 2)
    bar = "Drag distractors with NO flame | S save crops | U undo | C clear | N/P next/prev | Q quit"
    cv2.rectangle(view, (0,0), (view.shape[1], 58), (0,0,0), -1)
    cv2.putText(view, f"{index+1}/{len(files)} {files[index].name}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    cv2.putText(view, f"{bar} | boxes: {len(boxes)}", (10,47), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    cv2.imshow(WINDOW, view)


def save_crops():
    global boxes
    if not boxes:
        print("No boxes selected.")
        return
    LABELS.mkdir(parents=True, exist_ok=True)
    saved = 0
    h, w = frame.shape[:2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for n, (x1, y1, x2, y2) in enumerate(boxes, 1):
        # Add context around the selected distractor while staying inside the source image.
        pad_x, pad_y = int((x2-x1)*0.35), int((y2-y1)*0.35)
        left, top = max(0, x1-pad_x), max(0, y1-pad_y)
        right, bottom = min(w, x2+pad_x), min(h, y2+pad_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        stem = f"hardneg_{files[index].stem}_{stamp}_{n}"
        out_img, out_lab = IMAGES / f"{stem}.jpg", LABELS / f"{stem}.txt"
        if not cv2.imwrite(str(out_img), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not save {out_img}")
        out_lab.write_text("", encoding="utf-8")
        print(f"Saved empty-label negative crop: {out_img.name}")
        saved += 1
    print(f"Saved {saved} crop(s). Review that each crop contains no flame.")
    boxes = []

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, mouse)
load_current()
print(f"Reviewing {len(files)} captured positive images from {IMAGES}")
print("Negative crops are saved into the existing train/images and train/labels folders; originals are unchanged.")
while True:
    display()
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("u") and boxes:
        boxes.pop()
    elif key == ord("c"):
        boxes.clear()
    elif key == ord("s"):
        save_crops()
    elif key == ord("n") and index < len(files)-1:
        index += 1
        load_current()
    elif key == ord("p") and index > 0:
        index -= 1
        load_current()
cv2.destroyAllWindows()
