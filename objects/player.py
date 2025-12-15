import numpy as np

from objects.trackable_object import TrackableObject


class Player(TrackableObject):

    def __init__(self, object_id: int, previous_positions = None):
        super().__init__(object_id, previous_positions)
        self.possession_time = 0
        self.avg_speed = 0
        self.num_shots = 0
        self.num_goals = 0

    def get_distance(self, time_index, box2):
        """
        Getting the distance between two boxes centers
        """
        box1 = self.previous_positions[time_index]
        box_center = lambda box: (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
        box1_center, box2_center = box_center(box1), box_center(box2)
        return np.hypot(box1_center[0] - box2_center[0], box1_center[1] - box2_center[1])

