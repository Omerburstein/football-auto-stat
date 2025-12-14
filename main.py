
import cv2
import numpy as np
from ultralytics import YOLO

from possession import add_new_players_to_possession, closest_player

VIDEO_PATH = r"tests_videos\A1606b0e6_0 (27).mp4"

# load model
model = YOLO(r"weights\best_player_detection.pt")

video = cv2.VideoCapture(VIDEO_PATH)

results = model.track(VIDEO_PATH, stream=True, persist=True, save=True)


players_possession = {}
players_ids = set()
for result in results:
    ids = result.boxes.id.numpy().astype(int)
    boxes = result.boxes.xyxy.numpy().astype(int)
    class_ids = result.boxes.cls.numpy().astype(int)
    players_ids.update(ids[class_ids == 1])
    ball_box = boxes[class_ids == 0][0]
    players_possession = add_new_players_to_possession(players_possession, players_ids)
    
    # Add possession to a player
    players_possession[closest_player(players_ids, boxes[class_ids == 1], ball_box)] += 1






# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

    # frame = cv2.resize(frame, (1020, 600))
    # results = model.track(frame, persist=True, classes=[0, 32])
    # for result in results:


    # cv2.imshow("frame", frame)





