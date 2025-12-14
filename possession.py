import numpy as np


def add_new_players_to_possession(players_possession, players_ids):
    for player_id in players_ids:
        if player_id not in players_possession:
            players_possession[id] = 0
    return players_possession


def closest_player(players_ids, players_boxes, ball_position):
    min_dist = np.inf
    closest_index = 0
    for i, box in enumerate(players_boxes):
        current_dist = get_distance(box, ball_position)
        if min_dist < current_dist:
            min_dist = current_dist
            closest_index = i
    return players_ids[closest_index]

def get_distance(box1, box2):
    """
    Getting the distance between two boxes centers
    """
    box_center = lambda box: (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
    box1_center = box_center(box1)
    box2_center = box_center(box2)
    return np.hypot(box1_center[0] - box2_center[0], box1_center[1] - box2_center[1])
