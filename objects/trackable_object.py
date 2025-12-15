from typing import List

import pandas as pd


class TrackableObject:

    def __init__(self, object_id: int, previous_positions:List[int]=None):
        self.object_id = object_id
        self.previous_positions = []
        if previous_positions is not None:
            self.previous_positions = previous_positions

    def add_previous_position(self, positions: List[int]):
        self.previous_positions.extend(positions)

    def interpolate_positions(self):
        positions = [x.get(1, {}).get("bbox", []) for x in self.previous_positions]
        df_positions = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_positions = df_positions.bfill()
        return df_positions

