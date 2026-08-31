#!/usr/bin/env bash
# アプリを起動する。必ずこのリポジトリの .venv を使う
# （PATH 上の streamlit は別環境のことがあり、依存が揃っていない）
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  echo "エラー: .venv がありません。先にセットアップしてください:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec .venv/bin/streamlit run app.py "$@"
