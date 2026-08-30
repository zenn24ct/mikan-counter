"""Sequential video reading and frame sampling.

Frames are read one by one with cv2.VideoCapture and yielded lazily:
a several-hundred-MB video must never be loaded into memory at once.
No Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    total_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Read metadata without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def planned_frame_count(total_frames: int, every: int) -> int:
    """How many frames will be processed for a given sampling step."""
    every = max(1, int(every))
    if total_frames <= 0:
        return 0
    return (total_frames + every - 1) // every


def iter_frames(path: str | Path, every: int = 5) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) for every Nth frame, decoding sequentially."""
    every = max(1, int(every))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


def blur_score(frame: np.ndarray) -> float:
    """Variance of the Laplacian. Lower means blurrier."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
