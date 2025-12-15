
import cv2
from ultralytics import YOLO
from typing import List

from Tracker import Tracker
from objects.ball import Ball

VIDEO_PATH = r"tests_videos\A1606b0e6_0 (27).mp4"

# load model
model = YOLO(r"weights\best_player_detection.pt")

video = cv2.VideoCapture(VIDEO_PATH)

results = model.track(VIDEO_PATH, stream=True, persist=True, save=True)

# list of all players
players_ids = set()
ball = Ball(0)
game_tracker = Tracker(players = [], ball=ball)

for time_index, result in enumerate(results):
    ids = result.boxes.id.numpy().astype(int)
    boxes = result.boxes.xyxy.numpy().astype(int)
    class_ids = result.boxes.cls.numpy().astype(int)

    # Add players positions
    players_ids.update(ids[class_ids == 1])

    # In case there are new players add them
    game_tracker.players = game_tracker.add_new_players(players_ids, time_index)
    for current_player in game_tracker.players:
        if current_player.object_id not in ids:
            current_player.previous_positions.append([])
        else:
            current_index = list(ids).index(current_player.object_id)
            current_player.previous_positions.append(boxes[current_index])

    # Add ball position
    ball_position = boxes[class_ids == 2]
    if ball_position.size > 0:
        game_tracker.ball.previous_positions.append(ball_position[0])
    else:
        game_tracker.ball.previous_positions.append([])






# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

    # frame = cv2.resize(frame, (1020, 600))
    # results = model.track(frame, persist=True, classes=[0, 32])
    # for result in results:


    # cv2.imshow("frame", frame)





