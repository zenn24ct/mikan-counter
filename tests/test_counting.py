"""src/counting.py の単体テスト（ダミーの検出結果で集計だけを確認する）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import counting  # noqa: E402

SAMPLE = [(0, 3), (5, 7), (10, 7), (15, 1), (20, 0)]


def test_normalize_accepts_tuples_and_dicts():
    items = counting.normalize([(0, 3), {"frame_index": 5, "count": 7}])
    assert items == [counting.FrameCount(0, 3), counting.FrameCount(5, 7)]


def test_max_and_median():
    assert counting.max_count(SAMPLE) == 7
    assert counting.median_count(SAMPLE) == 3.0


def test_empty_input_is_zero_not_error():
    assert counting.max_count([]) == 0
    assert counting.median_count([]) == 0.0
    assert counting.peak_frame_index([]) is None


def test_peak_frame_index_takes_first_max():
    assert counting.peak_frame_index(SAMPLE) == 5


def test_top_frames_sorted_by_count_then_index():
    top = counting.top_frames(SAMPLE, n=3)
    assert [(fc.frame_index, fc.count) for fc in top] == [(5, 7), (10, 7), (0, 3)]


def test_unique_track_ids_ignores_none():
    assert counting.unique_track_ids([1, 2, 2, 3, None, None]) == 3
    assert counting.unique_track_ids([]) == 0


def test_summarize_without_tracking():
    summary = counting.summarize(
        SAMPLE, total_frames=100, fps=30.0, elapsed_sec=2.5, stage="green"
    )
    assert summary.max_count == 7
    assert summary.median_count == 3.0
    assert summary.processed_frames == 5
    assert summary.total_frames == 100
    assert summary.fps == 30.0
    assert summary.unique_track_ids is None
    assert summary.has_detection is True


def test_summarize_with_tracking_and_no_detection():
    summary = counting.summarize(
        [(0, 0), (5, 0)],
        total_frames=10,
        fps=30.0,
        elapsed_sec=0.1,
        stage="colored",
        track_ids=[1, 1, 4],
    )
    assert summary.max_count == 0
    assert summary.has_detection is False
    assert summary.unique_track_ids == 2


def test_to_csv_with_and_without_stage():
    csv_text = counting.to_csv([(0, 3), (5, 7)])
    assert csv_text.splitlines()[0] == "frame_index,count"
    assert csv_text.splitlines()[1] == "0,3"

    staged = counting.to_csv([(0, 3)], stage="green")
    assert staged.splitlines()[0] == "frame_index,count,stage"
    assert staged.splitlines()[1] == "0,3,green"


@pytest.mark.parametrize("counts,expected", [([(0, 1)], 1.0), ([(0, 1), (1, 2)], 1.5)])
def test_median_of_even_and_odd_lengths(counts, expected):
    assert counting.median_count(counts) == expected
