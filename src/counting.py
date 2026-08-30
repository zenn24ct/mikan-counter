"""Aggregation of per-frame detection counts.

Pure functions only: no Streamlit, no OpenCV, no model objects.
Everything here must be testable with plain Python lists.
"""

from __future__ import annotations

import csv
import io
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FrameCount:
    """Number of detections found in one processed frame."""

    frame_index: int
    count: int


@dataclass(frozen=True)
class CountSummary:
    """Aggregated result for one video / one setting."""

    max_count: int
    median_count: float
    processed_frames: int
    total_frames: int
    fps: float
    elapsed_sec: float
    stage: str
    unique_track_ids: int | None = None
    peak_frame_index: int | None = None

    @property
    def has_detection(self) -> bool:
        return self.max_count > 0


def normalize(frame_counts: Iterable) -> list[FrameCount]:
    """Accept FrameCount, (index, count) tuples or dicts and return FrameCount list."""
    out: list[FrameCount] = []
    for item in frame_counts:
        if isinstance(item, FrameCount):
            out.append(item)
        elif isinstance(item, dict):
            out.append(FrameCount(int(item["frame_index"]), int(item["count"])))
        else:
            index, count = item
            out.append(FrameCount(int(index), int(count)))
    return out


def max_count(frame_counts: Iterable) -> int:
    counts = [fc.count for fc in normalize(frame_counts)]
    return max(counts) if counts else 0


def median_count(frame_counts: Iterable) -> float:
    counts = [fc.count for fc in normalize(frame_counts)]
    return float(statistics.median(counts)) if counts else 0.0


def peak_frame_index(frame_counts: Iterable) -> int | None:
    """Frame index of the first frame reaching the maximum count."""
    items = normalize(frame_counts)
    if not items:
        return None
    best = max(items, key=lambda fc: fc.count)
    return best.frame_index


def top_frames(frame_counts: Iterable, n: int = 4) -> list[FrameCount]:
    """Frames with the highest counts (ties broken by frame order)."""
    items = normalize(frame_counts)
    ranked = sorted(items, key=lambda fc: (-fc.count, fc.frame_index))
    return ranked[:n]


def unique_track_ids(track_ids: Iterable) -> int:
    """Number of distinct tracker IDs, ignoring None."""
    return len({tid for tid in track_ids if tid is not None})


def summarize(
    frame_counts: Iterable,
    *,
    total_frames: int,
    fps: float,
    elapsed_sec: float,
    stage: str,
    track_ids: Sequence | None = None,
) -> CountSummary:
    items = normalize(frame_counts)
    return CountSummary(
        max_count=max_count(items),
        median_count=median_count(items),
        processed_frames=len(items),
        total_frames=int(total_frames),
        fps=float(fps),
        elapsed_sec=float(elapsed_sec),
        stage=stage,
        unique_track_ids=None if track_ids is None else unique_track_ids(track_ids),
        peak_frame_index=peak_frame_index(items),
    )


def to_csv(frame_counts: Iterable, *, stage: str | None = None) -> str:
    """CSV text with a frame_index / count column (plus stage when given)."""
    items = normalize(frame_counts)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ["frame_index", "count"]
    if stage is not None:
        header.append("stage")
    writer.writerow(header)
    for fc in items:
        row = [fc.frame_index, fc.count]
        if stage is not None:
            row.append(stage)
        writer.writerow(row)
    return buffer.getvalue()
