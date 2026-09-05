#!/usr/bin/env bash
# DR-048 Prompt 8（選配）：A5 @ hier、**forward 順序**、十折。
#
#   python scripts/run_exp2.py --arms A5 --order main --arch hier --fold k --seeds k --tag sota
#
# 補上 SOTA 表最後一格空缺 —— 在此之前 forward 只有 flat，
# 因此「hier − flat」的配對只能在 reverse 下做。
#
# 與 scripts/sota_queue.sh 同一套規矩：可續跑（產物已存在即跳過，由 run_exp2 內建
# resume 負責）、逐 run 一份 log、單一 run 失敗不中斷整批、失敗清單另存。
set -u
cd "$(dirname "$0")/.."

TAG="${SOTA_TAG:-sota}"
LOGDIR="logs/sota"
FAILED="$LOGDIR/FAILED_hier_fwd.txt"
mkdir -p "$LOGDIR"
: > "$FAILED"

step=0
for k in 1 2 3 4 5 6 7 8 9 10; do
  step=$((step + 1))
  name="A5_hier_fwd_f$k"
  printf '[%s] #%02d %s\n' "$(date '+%F %T')" "$step" "$name"
  if ! python scripts/run_exp2.py --arms A5 --order main --arch hier \
        --fold "$k" --seeds "$k" --tag "$TAG" > "$LOGDIR/$name.log" 2>&1; then
    printf '    ❌ 失敗 → %s\n' "$LOGDIR/$name.log"
    echo "$name" >> "$FAILED"
  fi
done

n=$(wc -l < "$FAILED" | tr -d ' ')
printf '[%s] 佇列結束：%d 步，失敗 %s 個\n' "$(date '+%F %T')" "$step" "$n"
[ "$n" -gt 0 ] && cat "$FAILED"
exit 0
