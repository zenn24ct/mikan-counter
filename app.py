"""みかん果実カウンタ（PC版プロトタイプ）

動画に写っている果実を数えるだけのアプリです。収量の予測は行いません。
UI はこのファイルのみが持ち、検出・集計ロジックは src/ にあります。
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import counting, detector, video  # noqa: E402

STAGE_LABELS = {
    detector.STAGE_GREEN: "緑果（摘果期）",
    detector.STAGE_COLORED: "着色果（収穫期）",
}
LABEL_TO_STAGE = {v: k for k, v in STAGE_LABELS.items()}

BUILTIN_MODELS = ["yolo11n.pt", "yolov8n.pt"]
CUSTOM_MODEL_LABEL = "カスタム（.pt のパスを指定）"
PREVIEW_COUNT = 4

st.set_page_config(page_title="みかん果実カウンタ", page_icon="🍊", layout="wide")


@st.cache_resource(show_spinner="モデルを読み込んでいます…")
def get_model(model_path: str):
    return detector.load_model(model_path)


def process_video(
    video_path: str,
    *,
    stage: str,
    model_path: str,
    conf: float,
    every: int,
    orange_only: bool,
    track: bool,
    imgsz: int,
) -> dict:
    """動画を逐次読みして検出・集計する。結果は session_state に入る辞書。"""
    info = video.probe(video_path)
    model = get_model(detector.model_path_for_stage(stage, model_path))
    device = detector.select_device()
    classes = detector.resolve_classes(model, orange_only)

    planned = video.planned_frame_count(info.total_frames, every)
    progress = st.progress(0.0, text="検出中…")

    frame_counts: list[counting.FrameCount] = []
    track_ids: list[int] = []
    previews: list[dict] = []
    started = time.time()

    for processed, (frame_index, frame) in enumerate(video.iter_frames(video_path, every), start=1):
        detections = detector.detect(
            model,
            frame,
            stage=stage,
            conf=conf,
            classes=classes,
            device=device,
            imgsz=imgsz,
            track=track,
        )
        count = len(detections)
        frame_counts.append(counting.FrameCount(frame_index, count))
        if track:
            track_ids.extend(d.track_id for d in detections if d.track_id is not None)

        # プレビューは上位 4 枚だけ保持する（全フレームは載せない）
        weakest = min((p["count"] for p in previews), default=-1)
        if len(previews) < PREVIEW_COUNT or count > weakest:
            annotated = detector.draw_detections(frame, detections)
            previews.append(
                {
                    "frame_index": frame_index,
                    "count": count,
                    "image": annotated[:, :, ::-1].copy(),  # BGR -> RGB
                }
            )
            previews.sort(key=lambda p: (-p["count"], p["frame_index"]))
            del previews[PREVIEW_COUNT:]

        if planned:
            progress.progress(min(processed / planned, 1.0), text=f"検出中… {processed}/{planned} フレーム")

    progress.empty()
    elapsed = time.time() - started

    summary = counting.summarize(
        frame_counts,
        total_frames=info.total_frames,
        fps=info.fps,
        elapsed_sec=elapsed,
        stage=stage,
        track_ids=track_ids if track else None,
    )
    return {
        "summary": summary,
        "frame_counts": frame_counts,
        "previews": previews,
        "settings": {
            "stage": stage,
            "model": model_path,
            "conf": conf,
            "every": every,
            "imgsz": imgsz,
            "orange_only": orange_only,
            "track": track,
            "device": device,
            "is_coco": detector.is_coco_model(model),
        },
    }


def render_result(result: dict) -> None:
    summary: counting.CountSummary = result["summary"]
    settings = result["settings"]

    st.subheader("結果")
    st.metric("フレームあたり最大検出数", f"{summary.max_count} 個")

    cols = st.columns(5)
    cols[0].metric("中央値", f"{summary.median_count:.1f}")
    cols[1].metric("処理フレーム数", f"{summary.processed_frames}")
    cols[2].metric("総フレーム数", f"{summary.total_frames}")
    cols[3].metric("fps", f"{summary.fps:.1f}")
    cols[4].metric("処理時間", f"{summary.elapsed_sec:.1f} 秒")

    st.caption(
        f"生育ステージ: {STAGE_LABELS[settings['stage']]} / モデル: {settings['model']} / "
        f"信頼度: {settings['conf']:.2f} / 間引き: {settings['every']} / "
        f"推論解像度: {settings['imgsz']} / デバイス: {settings['device']}"
    )

    if summary.unique_track_ids is not None:
        st.metric("ユニークID数（追跡）", f"{summary.unique_track_ids}")
        st.warning("ユニークID数は二重カウント除去が未検証の参考値です。")

    if not summary.has_detection:
        st.info("検出0件でした。")
        if settings["stage"] == detector.STAGE_GREEN and settings["is_coco"]:
            st.warning(
                "COCO事前学習モデルは着色前の未熟果を学習していません。"
                "自前データでの学習が必要です（scripts/train.py）。"
            )

    frame_counts = result["frame_counts"]
    if frame_counts:
        chart_df = pd.DataFrame(
            {"検出数": [fc.count for fc in frame_counts]},
            index=[fc.frame_index for fc in frame_counts],
        )
        chart_df.index.name = "フレーム番号"
        st.line_chart(chart_df)

    previews = result["previews"]
    if previews:
        st.subheader("検出フレーム（検出数の多い順）")
        for column, preview in zip(st.columns(len(previews)), previews):
            column.image(
                preview["image"],
                caption=f"フレーム {preview['frame_index']} / {preview['count']} 個",
                width="stretch",
            )

    st.download_button(
        "CSVをダウンロード",
        data=counting.to_csv(frame_counts, stage=settings["stage"]).encode("utf-8-sig"),
        file_name="mikan_counts.csv",
        mime="text/csv",
    )


st.title("🍊 みかん果実カウンタ")
st.caption("動画に写っている果実を数えるアプリです。収量の予測は行いません。")

with st.sidebar:
    st.header("設定")
    stage_label = st.radio("生育ステージ", list(STAGE_LABELS.values()), index=0)
    stage = LABEL_TO_STAGE[stage_label]

    model_choice = st.selectbox("モデル", BUILTIN_MODELS + [CUSTOM_MODEL_LABEL], index=0)
    if model_choice == CUSTOM_MODEL_LABEL:
        model_path = st.text_input("カスタムモデルのパス", value="models/best.pt")
    else:
        model_path = model_choice
    is_builtin_coco = model_choice in BUILTIN_MODELS

    if is_builtin_coco:
        class_mode = st.radio("検出クラス", ["orange(49) のみ", "全クラス"], index=0)
        orange_only = class_mode == "orange(49) のみ"
    else:
        orange_only = False
        st.caption("カスタムモデルでは全クラスを検出します。")

    conf = st.slider("信頼度しきい値", 0.05, 0.9, 0.25, 0.05)
    every = st.number_input("フレーム間引き（Nフレームごとに1枚）", min_value=1, max_value=120, value=5, step=1)
    imgsz = st.selectbox(
        "推論解像度",
        [640, 960, 1280],
        index=0 if is_builtin_coco else 1,
        help="学習時と同じ解像度を選ぶ。models/best.pt は 960 で学習しています。",
    )
    track = st.checkbox("追跡（ByteTrack）", value=False)

    if stage == detector.STAGE_GREEN and is_builtin_coco:
        st.info(
            "COCO事前学習モデルは着色前の未熟果を学習していません。"
            "緑果では検出0件になる想定です（scripts/train.py で自前学習）。"
        )

uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded is not None:
    st.video(uploaded)

if st.button("検出を実行", type="primary", disabled=uploaded is None):
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        st.session_state["result"] = process_video(
            tmp_path,
            stage=stage,
            model_path=model_path,
            conf=conf,
            every=int(every),
            orange_only=orange_only,
            track=track,
            imgsz=int(imgsz),
        )
        st.session_state["result"]["video_name"] = uploaded.name
    except Exception as exc:  # モデル未取得・破損動画などをUIで見せる
        st.session_state.pop("result", None)
        st.error(f"処理に失敗しました: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

if "result" in st.session_state:
    st.divider()
    st.caption(f"対象動画: {st.session_state['result'].get('video_name', '-')}")
    render_result(st.session_state["result"])
