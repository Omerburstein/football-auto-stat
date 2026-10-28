from typing import List

import numpy as np

from objects.trackable_object import TrackableObject


class Player(TrackableObject):

    def __init__(self, object_id: int, previous_positions:List[int] = None):
        super().__init__(object_id, previous_positions)
        self.possession_time = 0
        self.avg_speed = 0
        self.num_shots = 0
        self.num_goals = 0

