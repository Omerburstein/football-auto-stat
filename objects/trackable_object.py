from typing import List

import numpy as np
import pandas as pd

from consts import TIME_BETWEEN_TWO_FRAMES


class TrackableObject:

    def __init__(self, object_id: int, previous_positions:List[List[int]]=None):
        self.object_id = object_id
        self.previous_positions = []
        if previous_positions is not None:
            self.previous_positions = previous_positions

    def add_previous_position(self, positions: List[int]):
        self.previous_positions.extend(positions)

    def interpolate_positions(self):
        df_positions = pd.DataFrame(self.previous_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_positions = df_positions.bfill()
        return df_positions.values.tolist()

    def get_distance(self, time_index, box2):
        """
        Getting the distance between two boxes centers
        """
        box1 = self.previous_positions[time_index]
        box_center = lambda box: (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
        box1_center, box2_center = box_center(box1), box_center(box2)
        return np.hypot(box1_center[0] - box2_center[0], box1_center[1] - box2_center[1])

    def calc_speed(self, time_index: int) -> float:
        if time_index < 0:
            return 0
        if time_index == 0:
            time_index = 1
        distance_in_pixels = self.get_distance(time_index, self.previous_positions[time_index - 1])
        return distance_in_pixels / TIME_BETWEEN_TWO_FRAMES