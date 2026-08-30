# mikan-counter

温州みかんの樹を撮った動画から、**動画に写っている果実を数える** PC版プロトタイプ（卒業研究用）。

> **このアプリは収量を予測しません。動画中に観測可能な果実を数えるだけです。**
> 収穫実測値（正解データ）を持っていないため、収量・予測に類する出力は意図的に持たせていません。

## できること

- 動画（mp4 / mov / avi）をアップロードして YOLO で果実を検出する
- 生育ステージ（`緑果（摘果期）` / `着色果（収穫期）`）を切り替えて、同じ動画・同じ設定で条件を比較する
- フレームあたり最大検出数を主指標として表示し、フレーム番号 × 検出数を CSV で出力する

素の COCO 事前学習モデル（`yolo11n.pt` / `yolov8n.pt`）は、着色果ならある程度検出できますが、
緑果ではほぼ検出できません。**これは不具合ではなく想定どおりの挙動で、この差そのものがベースラインの比較対象です。**
緑果を数えるには自前アノテーションでの学習（`scripts/train.py`）が必要です。

## セットアップ

```bash
cd mikan-counter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

起動:

```bash
streamlit run app.py
```

ブラウザで開いたら、サイドバーで設定を選び、動画をアップロードして「検出を実行」を押します。
モデル（`yolo11n.pt` など）は初回実行時に ultralytics が自動ダウンロードします（要ネットワーク）。

デバイスは `mps` → `cuda` → `cpu` の順で自動選択します（Apple Silicon の Mac では `mps`）。

## サイドバーの設定

| 項目 | 内容 |
| --- | --- |
| 生育ステージ | `緑果（摘果期）` / `着色果（収穫期）`。結果と CSV に記録される |
| モデル | `yolo11n.pt`（既定） / `yolov8n.pt` / カスタム `.pt` のパス |
| 検出クラス | COCOモデルのとき `orange(49)` のみ / 全クラス。カスタムモデルは常に全クラス |
| 信頼度しきい値 | 0.05〜0.9（既定 0.25） |
| フレーム間引き | Nフレームごとに1枚処理（既定 5） |
| 追跡 | ByteTrack による追跡（既定 OFF）。ユニークID数は**二重カウント除去が未検証の参考値** |

## アノテーションと学習

1. フレームを切り出す

```bash
python scripts/extract_frames.py --video data/tree01.mp4 --out data/frames/tree01 --every 10 --max 200
```

`--skip-blurry` を付けるとラプラシアン分散でボケたフレームを除外します（しきい値は `--blur-threshold`、既定 100.0）。
ファイル名は `{動画名}_{フレーム番号:06d}.jpg` なので、後からどの動画のどのフレームか追えます。

2. アノテーションする

[Label Studio](https://labelstud.io/) または [CVAT](https://www.cvat.ai/) で `data/frames/` の画像を読み込み、
矩形（bounding box）で果実を囲みます。クラスは 2 つ:

- `green_fruit` — 着色前の緑果（摘果期）
- `colored_fruit` — 着色済みの果実（収穫期）

見た目が大きく違うので、最初から分けて付けます（学習時に 1 クラスへ統合はできますが、逆はできません）。

エクスポートは **YOLO 形式**（Label Studio: "YOLO"、CVAT: "YOLO 1.1"）を選び、
`data/dataset/` に次の構成で配置します。

```
data/dataset/
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```

3. データセット定義を作る

```bash
cp data/mikan.yaml.example data/mikan.yaml
```

4. 学習する

```bash
python scripts/train.py --data data/mikan.yaml --model yolo11n.pt --epochs 100 --imgsz 640
```

学習後、`best.pt` が `models/best.pt` にコピーされます。
アプリのサイドバーで「カスタム」を選び、`models/best.pt` を指定すると使えます。

## テスト

```bash
python -m pytest tests/ -q
```

## 構成

```
mikan-counter/
├── app.py                  # Streamlit本体（UIはここだけ）
├── src/
│   ├── detector.py         # YOLO推論のラッパ（Streamlit非依存）
│   ├── video.py            # 動画の逐次読み込み・フレームサンプリング
│   └── counting.py         # 集計ロジック（純粋関数）
├── scripts/
│   ├── extract_frames.py   # アノテーション用フレーム切り出しCLI
│   └── train.py            # 自前データでの学習CLI
├── data/                   # .gitignore対象（動画・切り出し画像）
├── models/                 # .gitignore対象（.ptファイル）
├── tests/
├── requirements.txt
├── README.md
└── CLAUDE.md
```

`src/` は Streamlit に依存させていません（後でスマホアプリへ移植するため）。UI は `app.py` だけが持ちます。

## 補足

- アップロード上限は `.streamlit/config.toml` で 2GB に引き上げています（Streamlit の既定は 200MB）。
- 追跡（ByteTrack）には `lap` が必要です。`requirements.txt` に含めています。
- `ultralytics` は依存として非 headless の `opencv-python` を引き込むため、インストール後に
  `requirements.txt` 末尾のコマンドで headless 版へ戻してください。
