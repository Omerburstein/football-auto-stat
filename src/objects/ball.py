from typing import List

from objects.trackable_object import TrackableObject


class Ball(TrackableObject):

    def __init__(self, object_id, previous_positions: List[int]=None):
        super().__init__(object_id, previous_positions)
