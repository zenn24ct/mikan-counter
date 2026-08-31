"""Thin wrapper around Ultralytics YOLO inference.

No Streamlit dependency: this module only knows about frames (numpy arrays),
a growth stage label and detection results, so it can be reused later by the
mobile version.

`ultralytics` and `torch` are imported lazily inside functions so that the app
can start (and show a helpful message) even before the packages are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

# Growth stages. The value is what gets written to the CSV, so keep it stable.
STAGE_GREEN = "green"      # thinning season (Aug-Sep), fruit still green
STAGE_COLORED = "colored"  # harvest season, fruit coloured yellow/orange
STAGES = (STAGE_GREEN, STAGE_COLORED)

# COCO class id used by the pre-trained models.
COCO_ORANGE_CLASS_ID = 49

DEFAULT_TRACKER = "bytetrack.yaml"


@dataclass(frozen=True)
class Detection:
    """One detected box in one frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""
    track_id: int | None = None


def select_device() -> str:
    """Pick the best available device: mps -> cuda -> cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def model_path_for_stage(stage: str, model_path: str) -> str:
    """Resolve which weights to use for a growth stage.

    Today both stages share one model, but the stage is threaded through the API
    so a stage-specific model (green_fruit / colored_fruit) can be plugged in
    later without touching the UI.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return model_path


def load_model(model_path: str) -> Any:
    """Load a YOLO model. Raises RuntimeError with a readable message on failure."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ultralytics is not installed. run: pip install -r requirements.txt"
        ) from exc
    return YOLO(model_path)


def class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def is_coco_model(model: Any) -> bool:
    """True when the model still uses the 80-class COCO vocabulary."""
    names = class_names(model)
    return len(names) == 80 and names.get(COCO_ORANGE_CLASS_ID) == "orange"


def resolve_classes(model: Any, orange_only: bool) -> list[int] | None:
    """Class filter passed to YOLO. None means 'keep every class'."""
    if orange_only and is_coco_model(model):
        return [COCO_ORANGE_CLASS_ID]
    return None


def detect(
    model: Any,
    frame: np.ndarray,
    *,
    stage: str = STAGE_GREEN,
    conf: float = 0.25,
    classes: Sequence[int] | None = None,
    device: str | None = None,
    imgsz: int | None = None,
    track: bool = False,
    tracker: str = DEFAULT_TRACKER,
) -> list[Detection]:
    """Run inference on a single BGR frame and return the detections.

    `stage` is currently informational (see model_path_for_stage) but is part of
    the signature on purpose: stage-specific models are the next step.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")

    kwargs: dict[str, Any] = {
        "conf": conf,
        "verbose": False,
    }
    if classes is not None:
        kwargs["classes"] = list(classes)
    if device:
        kwargs["device"] = device
    if imgsz:
        # must match the size the model was trained at, or small fruit is lost
        kwargs["imgsz"] = int(imgsz)

    if track:
        results = model.track(frame, persist=True, tracker=tracker, **kwargs)
    else:
        results = model.predict(frame, **kwargs)

    return parse_results(results, class_names(model))


def parse_results(results: Any, names: dict[int, str] | None = None) -> list[Detection]:
    """Convert an ultralytics Results list into plain Detection objects."""
    names = names or {}
    detections: list[Detection] = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            class_id = int(clss[i])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(confs[i]),
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
    return detections


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Return a copy of the frame with boxes and labels drawn (BGR)."""
    canvas = frame.copy()
    color = (0, 200, 255)
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, p1, p2, color, 2)
        label = f"{det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        cv2.putText(
            canvas, label, (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return canvas
