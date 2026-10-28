"""Detection and identity tracking of players from fixed camera(s).

Stages: detect + online track (short tracklets) -> appearance embeddings -> offline global
identity assignment against an enrolled, named roster -> per-frame GameState.
Entry point: `py -3.13 -m tracking.pipeline <match.yaml>`.
"""
