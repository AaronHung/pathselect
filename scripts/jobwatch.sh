#!/usr/bin/env bash
# 看長跑佇列的狀態。從任何一個新開的終端機都能跑，不需要接回原本的 session。
#
#   bash scripts/jobwatch.sh        一次性快照
#   bash scripts/jobwatch.sh -f     快照後接著即時跟著現在這個 run 的輸出（Ctrl-C 離開）
#
# ⚠️ 腳本內一律用 /bin/ls —— 互動 shell 的 `ls` 是 eza 別名（帶圖示字元），
#    塞進 $( ) 會得到帶控制碼的檔名，tail 會說找不到檔案。
set -u
cd "$(dirname "$0")/.."

PIDF=logs/sota_queue.pid
MAIN=logs/sota_queue.log
RUNDIR=logs/sota
TOTAL=62          # 佇列總步數：A5 flat 10 + A5 hier 10 + zeroshot 10 + opcm 1
                  #             + （選配）A5 fwd 10 + A1/A3 20 + 產表 1

hr() { printf '─%.0s' $(seq 1 60); echo; }

# ⚠️ `pgrep -f sota_queue.sh` 會命中**兩個**行程：佇列本身，以及 caffeinate
#    ——因為 caffeinate 的命令列是 `caffeinate -is bash scripts/sota_queue.sh`，
#    整串裡也含這個字。所以這裡用 comm 過濾，只取真正的 bash 那一個。
QPID=$(pgrep -f sota_queue.sh 2>/dev/null | while read -r p; do
         [ "$(ps -o comm= -p "$p" | tr -d ' ')" = "bash" ] && echo "$p"; done | head -1)

hr
if [ -n "${QPID:-}" ]; then
  EL=$(ps -o etime= -p "$QPID" | tr -d ' ')
  PGID=$(ps -o pgid= -p "$QPID" | tr -d ' ')
  echo "✅ 佇列在跑   pid=$QPID   已跑 $EL"
  echo "   要全部停掉：kill -TERM -$PGID   ← 負號＝殺整個 process group"
  echo "   （只 kill $QPID 的話，正在算的 python 會變孤兒繼續跑）"
else
  echo "❌ 佇列沒在跑（可能已完成，或被中斷）"
  [ -f "$PIDF" ] && echo "   最後記錄的 pid：$(cat "$PIDF")"
fi

pgrep -x caffeinate > /dev/null \
  && echo "☕ caffeinate 在擋睡眠" \
  || echo "⚠️  沒有 caffeinate —— 闔蓋可能會暫停"

# 正在算的那個 python（每一步會換一個新的）
if [ -n "${QPID:-}" ]; then
  CHILD=$(pgrep -P "$QPID" -x python 2>/dev/null | head -1)
  [ -n "$CHILD" ] && echo "🐍 目前這步的 python pid=$CHILD" &&
    ps -o command= -p "$CHILD" | cut -c1-95 | sed 's/^/   /'
fi

hr
if [ -f "$MAIN" ]; then
  DONE=$(grep -c '^\[' "$MAIN")
  echo "進度：第 $DONE / $TOTAL 步"
  echo "現在這步：$(grep '^\[' "$MAIN" | tail -1)"

  # ETA：用已完成各步的實際間隔推估
  if [ "$DONE" -ge 2 ]; then
    python3 - "$MAIN" "$TOTAL" <<'PY'
import sys, re, datetime
lines = [l for l in open(sys.argv[1]) if l.startswith('[')]
total = int(sys.argv[2])
ts = [datetime.datetime.strptime(re.match(r'\[(.*?)\]', l).group(1), '%Y-%m-%d %H:%M:%S')
      for l in lines]
gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
if gaps:
    avg = sum(gaps) / len(gaps)
    left = total - len(ts)
    eta = datetime.datetime.now() + datetime.timedelta(seconds=avg * left)
    print(f"每步平均 {avg/60:.1f} 分；剩 {left} 步 → 約 {avg*left/3600:.1f} 小時，"
          f"預計 {eta:%m-%d %H:%M} 完成")
    print("（含選配段。只算必做的前 31 步請自行折半。）")
PY
  fi
fi

hr
LATEST=$(/bin/ls -t "$RUNDIR"/*.log 2>/dev/null | head -1)
if [ -n "${LATEST:-}" ]; then
  echo "目前這個 run：$LATEST"
  tail -3 "$LATEST" | sed 's/^/   /'
fi

if [ -s "$RUNDIR/FAILED.txt" ]; then
  hr; echo "❌ 失敗的 run："; sed 's/^/   /' "$RUNDIR/FAILED.txt"
fi
hr

if [ "${1:-}" = "-f" ] && [ -n "${LATEST:-}" ]; then
  echo "跟著 $LATEST（Ctrl-C 離開，不會影響佇列）"; echo
  tail -f "$LATEST"
fi
