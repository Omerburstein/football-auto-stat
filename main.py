
import cv2
from ultralytics import YOLO
from typing import List

from Tracker import Tracker
from consts import BALL_CLASS_ID, PLAYER_CLASS_ID
from objects.ball import Ball

SECOND_VIDEO = "A1606b0e6_0 (27).mp4"
VIDEO_PATH = r"tests_videos\08fd33_4.mp4"

# load model
model = YOLO(r"weights\best_both.pt")

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
    players_ids.update(ids[class_ids == BALL_CLASS_ID])

    # In case there are new players add them
    game_tracker.players = game_tracker.add_new_players(players_ids, time_index)
    game_tracker.add_players_positions(ids, boxes)

    # Add ball position
    ball_position = boxes[class_ids == PLAYER_CLASS_ID]
    if ball_position.size > 0:
        game_tracker.ball.previous_positions.append(ball_position[0])
    else:
        game_tracker.ball.previous_positions.append([])


game_tracker.ball.interpolate_positions()
game_tracker.find_possession()



# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

    # frame = cv2.resize(frame, (1020, 600))
    # results = model.track(frame, persist=True, classes=[0, 32])
    # for result in results:


    # cv2.imshow("frame", frame)





