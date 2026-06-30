from typing import TypedDict

class Track(TypedDict):
    frame_idx: list[int]
    x: list[float]
    y: list[float]
    pred_x: list[float]
    pred_y: list[float]
    size_frame: float
    size_sec: float
    dist_x: float
    dist_y: float
    first_frame: int
    last_frame: int

class Detection(TypedDict):
    frame: int
    visible: int
    x: float
    y: float
