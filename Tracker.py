from typing import List

import numpy as np

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
            self.closest_player(time_index).possession_time += 1
        return [100*player.possession_time/game_length for player in self.players]


    def closest_player(self, time_index) -> Player:
        min_dist = np.inf
        closest_player = self.players[0]
        for i, player in enumerate(self.players):
            current_dist = player.get_distance(time_index, self.ball.previous_positions[time_index])
            if min_dist < current_dist:
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
