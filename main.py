
import cv2
from ultralytics import YOLO

VIDEO_PATH = r"tests_videos\A1606b0e6_0 (27).mp4"

# load model
model = YOLO("yolo12s.pt")
video = cv2.VideoCapture(VIDEO_PATH)

results = model.track(VIDEO_PATH, stream=True, persist=True, save=True)

for result in results:
    ids = result.boxes.id.numpy().astype(int)
    boxes = result.boxes.xyxy.numpy().astype(int)
    class_ids = result.boxes.cls.numpy().astype(int)
# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

    # frame = cv2.resize(frame, (1020, 600))
    # results = model.track(frame, persist=True, classes=[0, 32])
    # for result in results:


    # cv2.imshow("frame", frame)





