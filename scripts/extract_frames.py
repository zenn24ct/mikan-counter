#!/usr/bin/env python3
"""Extract frames from a video for annotation.

Example:
    python scripts/extract_frames.py --video data/tree01.mp4 --out data/frames/tree01 \
        --every 10 --max 200 --skip-blurry --blur-threshold 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import video  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動画からアノテーション用フレームを切り出す")
    parser.add_argument("--video", required=True, help="入力動画のパス")
    parser.add_argument("--out", required=True, help="出力ディレクトリ")
    parser.add_argument("--every", type=int, default=10, help="Nフレームごとに1枚保存（既定: 10）")
    parser.add_argument("--max", type=int, default=None, help="保存する最大枚数")
    parser.add_argument("--skip-blurry", action="store_true", help="ボケたフレームを保存しない")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="ラプラシアン分散のしきい値。これ未満をボケとみなす（既定: 100.0）",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG品質（既定: 95）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    saved = 0
    skipped_blurry = 0
    for frame_index, frame in video.iter_frames(video_path, args.every):
        if args.max is not None and saved >= args.max:
            break
        if args.skip_blurry and video.blur_score(frame) < args.blur_threshold:
            skipped_blurry += 1
            continue
        out_path = out_dir / f"{stem}_{frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        saved += 1

    print(f"saved {saved} frames to {out_dir}")
    if args.skip_blurry:
        print(f"skipped {skipped_blurry} blurry frames (threshold {args.blur_threshold})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
