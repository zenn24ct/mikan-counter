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
pip install -r requirements.txt      # テストも動かすなら requirements-dev.txt
```

起動:

```bash
./run.sh
```

`streamlit run app.py` と直接打つと、PATH 上の別の Python 環境（Anaconda など）の streamlit が使われて
依存が足りずに落ちることがある。`run.sh` は必ずこのリポジトリの `.venv` を使う。

ブラウザで開いたら、サイドバーで設定を選び、動画をアップロードして「検出を実行」を押します。
モデル（`yolo11n.pt` など）は初回実行時に ultralytics が自動ダウンロードします（要ネットワーク）。

デバイスは `mps` → `cuda` → `cpu` の順で自動選択します（Apple Silicon の Mac では `mps`）。

## サイドバーの設定

| 項目 | 内容 |
| --- | --- |
| 生育ステージ | `緑果（摘果期）` / `着色果（収穫期）`。結果と CSV に記録される |
| モデル | `models/` に `.pt` があればそれが既定。COCOモデル（`yolo11n.pt` など）は `models/` が空のときだけ選択肢に出る |
| 信頼度しきい値 | 0.05〜0.9（既定 0.25） |
| フレーム間引き | Nフレームごとに1枚処理（既定 5） |
| 推論解像度 | 640 / 960 / 1280。**学習時と同じ値にする**。自前モデルは既定 960、COCOモデルは 640 |
| 追跡 | ByteTrack による追跡（既定 OFF）。ユニークID数は**二重カウント除去が未検証の参考値** |

結果画面では、処理した全フレームをスライダーで1枚ずつ送りながら検出枠を確認できる
（最大200枚。長い動画では等間隔に間引いて保持する）。加えて検出数の多い4枚を並べて表示する。

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

## Web版（常時アクセスできるURL）

本体は **Streamlit Community Cloud** にホストします。Streamlit は常駐サーバと WebSocket を使うため、
Vercel のようなサーバーレス環境では動きません（torch だけで 500MB 超あり、関数のサイズ上限も超えます）。

デプロイ手順（初回のみ、ブラウザ操作）:

1. https://share.streamlit.io にサインインする（GitHub / Google / メールのいずれか）
2. GitHub アカウントを連携する
   - 左上の「Workspaces ⚠」→「Connect GitHub account」→ GitHub で認証 →「Authorize streamlit」
3. private リポジトリからデプロイするので、追加の許可を与える
   - 左上の GitHub ユーザー名 →「Settings」→ 左の「Linked accounts」
   - "Source control" の「Connect here →」→「Authorize streamlit」
   - （リポジトリを public にする場合はこの手順は不要）
4. 右上の「Create app」→ GitHub からデプロイを選ぶ
   - Repository: `zenn24ct/mikan-counter`
   - Branch: `main`
   - Main file path: `app.py`
   - Advanced settings → Python version: **3.12**
   - App URL: `mikan-counter`（→ `https://mikan-counter.streamlit.app`）
5. Deploy を押す。初回は torch と ultralytics のインストールで 5〜10 分ほどかかる

サインインでエラーが出る場合は、https://share.streamlit.io/logout でサインアウトしてから、
別のプロバイダ（Google など）で入り直すと通ることがある。

以後は `main` に push するたびに自動で再デプロイされます。

Web版の制約:

- 無料枠の **CPU 推論**なので、ローカル（Apple Silicon の `mps`）より大幅に遅い。数十秒の短い動画で試すのが現実的
- メモリも限られるため、数百MBの動画はローカル版で処理する
- しばらくアクセスがないとスリープし、次回起動に数十秒かかる
- `packages.txt` は Community Cloud 用の apt パッケージ（`libgl1` など。ultralytics が非 headless の
  opencv-python を引き込むため必要）

## テスト

```bash
pip install -r requirements-dev.txt
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
- `ultralytics` は依存として非 headless の `opencv-python` を引き込むため、ローカルではインストール後に
  `requirements.txt` 末尾のコマンドで headless 版へ戻してください。
- Linux では torch を CPU 版（`+cpu`）に固定しています。CUDA 同梱版だとクラウドのディスクに収まりません。
