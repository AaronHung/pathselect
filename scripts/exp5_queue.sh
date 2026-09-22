#!/usr/bin/env bash
# DR-057 / exp5：M=64 備份實驗的工作佇列執行器（5090 pod 專用）。
#
# 設計：一個 run 結束就補下一個，不固定分波 —— 各設定的 wall-clock 差很多
# （B=4 比 B=16 快、esca 折比 brca 折快），固定分波會讓整批被最慢的那一個拖住。
#
# 佇列檔一行一個 run：  <tag>\t<額外參數>
# 已完成的 run（有 .rc 且為 0）會跳過，所以中斷後直接重跑本腳本即可續跑。
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
QUEUE=${1:-logs/exp5/queue.tsv}
WORKERS=${2:-10}
VENV=${VENV:-/workspace/venvs/ps5}

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 PATHSELECT_TORCH_THREADS=8
mkdir -p logs/exp5/runs outputs/exp5

# 佇列游標用 flock 保護 —— 多個 worker 同時取號時不能撞號
CURSOR=logs/exp5/.cursor
LOCK=logs/exp5/.lock
[ -f "$CURSOR" ] || echo 0 > "$CURSOR"
: > "$LOCK"

total=$(grep -vc '^\s*\(#\|$\)' "$QUEUE")
echo "[queue] $total 個 run，$WORKERS 路平行，venv=$VENV"
date +%s > logs/exp5/QUEUE.start

take_next () {                     # 印出下一行的行號；沒了就印空字串
  flock 9
  local i
  i=$(cat "$CURSOR")
  i=$((i + 1))
  echo "$i" > "$CURSOR"
  echo "$i"
}

worker () {
  local wid=$1
  while :; do
    local n line
    n=$(take_next 9<"$LOCK")
    line=$(grep -v '^\s*\(#\|$\)' "$QUEUE" | sed -n "${n}p")
    [ -z "$line" ] && break

    local tag args
    tag=$(printf '%s' "$line" | cut -f1)
    args=$(printf '%s' "$line" | cut -f2-)
    local d=logs/exp5/runs/$tag

    if [ -f "$d.rc" ] && [ "$(cat "$d.rc")" = "0" ]; then
      echo "[w$wid] $tag 已完成，跳過"
      continue
    fi

    mkdir -p "$(dirname "$d")"
    echo "[w$wid] ▶ $tag"
    date +%s > "$d.start"
    # shellcheck disable=SC2086
    "$VENV/bin/python" scripts/run_exp2.py $args > "$d.log" 2>&1
    local rc=$?
    echo "$rc" > "$d.rc"
    date +%s > "$d.end"
    echo "[w$wid] $([ $rc -eq 0 ] && echo ✅ || echo ❌ rc=$rc) $tag  $(( $(cat "$d.end") - $(cat "$d.start") ))s"
  done
}

for w in $(seq 1 "$WORKERS"); do worker "$w" & done
wait
date +%s > logs/exp5/QUEUE.end
ok=$(grep -l '^0$' logs/exp5/runs/*.rc 2>/dev/null | wc -l)
bad=$(grep -L '^0$' logs/exp5/runs/*.rc 2>/dev/null | wc -l)
echo "[queue] 完成：成功 $ok、失敗 $bad"
