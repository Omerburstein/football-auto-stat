from typing import List

import numpy as np

from consts import MIN_DISTANCE_TO_BALL, MIN_BALL_SPEED_FOR_SHOOT
from objects.ball import Ball
from objects.player import Player


class Tracker:
    def __init__(self, players: List[Player], ball: Ball):
        self.players = players
        self.ball = ball

    def find_possession(self):
        # Finding the possession for all the players all over the game
        game_length = len(self.ball.previous_positions)
        for time_index in range(len(self.ball.previous_positions)):
            closest_player = self.closest_player(time_index)
            if closest_player is not None:
                closest_player.possession_time += 1
        return [100*player.possession_time/game_length for player in self.players]

    def closest_player(self, time_index) -> Player:
        min_dist = np.inf
        closest_player = None
        for i, player in enumerate(self.players):
            current_dist = player.get_distance(time_index, self.ball.previous_positions[time_index])
            if min_dist < current_dist < MIN_DISTANCE_TO_BALL:
                min_dist = current_dist
                closest_player = player
        return closest_player

    def get_player_by_id(self, player_id: int) -> Player:
        for player in self.players:
            if player.object_id == player_id:
                return player
        return None

    def add_new_players(self, players_ids: set, time_index: int) -> list:
        # In case there are new players add them with previous positions of empty list
        for player_id in players_ids:
            if self.get_player_by_id(player_id) is None:
                self.players.append(Player(player_id, previous_positions=[[]] * time_index))
        return self.players

    def add_players_positions(self, all_ids, all_boxes):
        for current_player in self.players:
            if current_player.object_id not in all_ids:
                current_player.previous_positions.append([])
            else:
                current_index = list(all_ids).index(current_player.object_id)
                current_player.previous_positions.append(all_boxes[current_index])

    def detect_shots(self):
        for time_index in range(len(self.ball.previous_positions)):
            if self.ball.calc_speed(time_index) > MIN_BALL_SPEED_FOR_SHOOT:
                shooting_player = self.closest_player(time_index)
                shooting_player.num_shots += 1